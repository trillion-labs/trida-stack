"""Last-prompt hidden-state readout for one eager vLLM generation worker.

The serial extension requires exclusive execution; the concurrent extension
routes captures by native request ID. Both read before vocabulary projection,
with one generated token per prefix, no cache, no chunking, and no speculation.
"""
import os

CAPTURE_TICKET_KEY = "trida_capture_ticket"


class DecisionWorkerExtension:
    def trida_install_hidden_capture(self):
        """Install after model loading, through collective_rpc; never load weights."""
        model = self.model_runner.model
        if not getattr(self, "_trida_capture_installed", False):
            original = model.compute_logits
            self._trida_capture = None

            def compute_logits(hidden_states, *args, **kwargs):
                state = self._trida_capture
                if state is not None:
                    state["calls"] += 1
                    if hidden_states.ndim != 2 or hidden_states.shape[0] != 1:
                        state["error"] = "capture requires exactly one sampled hidden state"
                    elif state["calls"] == 1:
                        # Clone before the normal logits path can reuse its buffers.
                        state["hidden"] = hidden_states[0].detach().clone()
                return original(hidden_states, *args, **kwargs)

            model.compute_logits = compute_logits
            self._trida_capture_installed = True
        return {"pid": os.getpid(), "model_id": id(model), "model_class": type(model).__name__}

    def trida_arm_hidden_capture(self, ticket):
        if not getattr(self, "_trida_capture_installed", False):
            raise RuntimeError("hidden capture is not installed")
        if self._trida_capture is not None:
            raise RuntimeError("another hidden capture is still armed")
        self._trida_capture = {"ticket": ticket, "calls": 0, "hidden": None}
        return ticket

    def trida_take_hidden_capture(self, ticket):
        import torch
        state = self._trida_capture
        if state is None or state["ticket"] != ticket:
            raise RuntimeError("hidden capture ticket does not match")
        if state.get("error") or state["calls"] != 1 or state["hidden"] is None:
            raise RuntimeError(state.get("error", "capture did not observe exactly one model call"))
        hidden = state["hidden"].float().cpu()
        if not torch.isfinite(hidden).all():
            raise RuntimeError("captured hidden state is not finite")
        self._trida_capture = None
        return {"ticket": ticket, "hidden": hidden.tolist(), "calls": state["calls"],
                "pid": os.getpid(), "model_id": id(self.model_runner.model)}

    def trida_clear_hidden_capture(self, ticket):
        # Late cleanup for a cancelled request must never clear a newer capture.
        if self._trida_capture is not None and self._trida_capture["ticket"] == ticket:
            self._trida_capture = None
        return self._trida_capture is None


class ConcurrentDecisionWorkerExtension:
    """Request-indexed capture for one eager V2 worker, without core patches."""

    def trida_install_hidden_capture(self, capacity=4):
        runner = self.model_runner
        if runner.batch_sharder is not None or not callable(getattr(runner, "add_requests", None)):
            raise RuntimeError("concurrent capture requires an unsharded V2 worker; use concurrency 1")
        if not 1 <= capacity <= 4:
            raise ValueError("capture capacity must be between 1 and 4")
        if getattr(self, "_trida_concurrent_installed", False):
            if capacity != self._trida_capacity:
                raise RuntimeError("capture capacity cannot change after installation")
            return self.trida_capture_status()
        self._trida_capacity = capacity
        self._trida_states, self._trida_bindings = {}, {}
        self._trida_metrics = {"captured": 0, "late_ignored": 0, "failed": 0, "cleared_pending": 0,
                               "cancelled_bound": 0, "cancelled_after_model": 0,
                               "batch_sizes": {}, "mixed_batch_sizes": {}, "multi_capture_batch_sizes": {}}
        original_add, original_sample = runner.add_requests, runner.sample

        def add_requests(output):
            result = original_add(output)
            for req in output.scheduled_new_reqs:
                extra = req.sampling_params.extra_args if req.sampling_params else None
                ticket = (extra or {}).get(CAPTURE_TICKET_KEY)
                if ticket is None:
                    continue
                state = self._trida_states.get(ticket)
                if state is None:  # Cancelled before admission: never recreate it.
                    self._trida_metrics["late_ignored"] += 1
                    continue
                if state["native_id"] is not None or req.prompt_len != state["expected_len"]:
                    state["error"] = "native request binding or prompt length mismatch"
                    continue
                state["native_id"] = req.req_id
                self._trida_bindings[req.req_id] = ticket
            return result

        def sample(hidden, batch, *args, **kwargs):
            captures = 0
            for i, req_id in enumerate(batch.req_ids[:batch.num_reqs]):
                ticket = self._trida_bindings.get(req_id)
                state = self._trida_states.get(ticket)
                if state is None:
                    continue
                captures += 1
                state["calls"] += 1
                lo, hi = map(int, batch.cu_num_logits_np[i:i + 2])
                final_prefill = (int(batch.num_computed_prefill_tokens_np[i])
                                 + int(batch.num_scheduled_tokens[i]) >= int(batch.prefill_len_np[i]))
                if (hi - lo != 1 or batch.num_draft_tokens != 0 or not batch.is_prefilling_np[i]
                        or not final_prefill or state["calls"] != 1
                        or int(batch.prefill_len_np[i]) != state["expected_len"]):
                    state["error"] = "ambiguous capture row or non-final prefill"
                    continue
                state["hidden"] = hidden[batch.logits_indices[lo]].detach().clone()
                self._trida_metrics["captured"] += 1
            for key, enabled in (("batch_sizes", True),
                                 ("mixed_batch_sizes", 0 < captures < batch.num_reqs),
                                 ("multi_capture_batch_sizes", captures > 1)):
                if enabled:
                    counts = self._trida_metrics[key]
                    size = int(batch.num_reqs)
                    counts[size] = counts.get(size, 0) + 1
            return original_sample(hidden, batch, *args, **kwargs)

        runner.add_requests, runner.sample = add_requests, sample
        self._trida_concurrent_installed = True
        return self.trida_capture_status()

    def trida_arm_hidden_capture(self, ticket, expected_len):
        if ticket in self._trida_states or len(self._trida_states) >= self._trida_capacity:
            raise RuntimeError("duplicate ticket or capture capacity exceeded")
        if type(expected_len) is not int or expected_len < 1:
            raise ValueError("capture needs its expanded prompt length")
        self._trida_states[ticket] = {"native_id": None, "expected_len": expected_len,
                                      "calls": 0, "hidden": None}
        return ticket

    def trida_take_hidden_capture(self, ticket):
        import torch
        state = self._trida_states.get(ticket)
        if state is None or state.get("error") or state["calls"] != 1 or state["hidden"] is None:
            self._trida_metrics["failed"] += 1
            raise RuntimeError("capture missing, stale, or ambiguous")
        hidden = state["hidden"].float().cpu()
        if hidden.ndim != 1 or not torch.isfinite(hidden).all():
            raise RuntimeError("captured hidden state is invalid")
        result = {"ticket": ticket, "hidden": hidden.tolist(), "calls": state["calls"],
                  "native_id": state["native_id"], "expected_len": state["expected_len"],
                  "pid": os.getpid(), "model_id": id(self.model_runner.model)}
        state["consumed"] = True
        self.trida_clear_hidden_capture(ticket)
        return result

    def trida_clear_hidden_capture(self, ticket):
        state = self._trida_states.pop(ticket, None)
        if state is not None:
            if state["native_id"] is not None:
                self._trida_bindings.pop(state["native_id"], None)
            if not state.get("consumed"):
                self._trida_metrics["cleared_pending"] += 1
                self._trida_metrics["cancelled_bound"] += int(state["native_id"] is not None)
                self._trida_metrics["cancelled_after_model"] += int(state["calls"] > 0)
        return ticket not in self._trida_states

    def trida_capture_status(self):
        return {"pid": os.getpid(), "model_id": id(self.model_runner.model),
                "active": len(self._trida_states), "bindings": len(self._trida_bindings),
                **{k: dict(v) if isinstance(v, dict) else v for k, v in self._trida_metrics.items()}}
