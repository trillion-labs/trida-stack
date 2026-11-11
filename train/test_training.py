"""CPU checks for training orchestration; no checkpoint downloads or CUDA required."""
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from train import checkpoint, distributed
from train import train as trainer
from train.cli import build_parser
from train.data.loaders import TrainingLoaders
from train.optimizer import build_optimizer, get_lr


class ToyModel(torch.nn.Module):
    """Real autograd with a scalar weight and recorded FSDP sync decisions."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.events = []

    def set_requires_gradient_sync(self, enabled):
        self.events.append(('sync', enabled))

    def forward(self, ids, labels, mask, *, decision=None, **kwargs):
        if decision is not None:
            self.events.append(('decision', None))
            return 2 * self.weight.square(), {
                'decision_acc': 1.0, 'decision_questions': 1, 'decision_tokens': 4,
            }
        self.events.append(('text', kwargs))
        loss = self.weight.square()
        return loss, {'diff_loss': loss.detach(), 'ar_loss': loss.detach(), 'ntok': 2}


@pytest.mark.parametrize('mode', ['text', 'mixed', 'decision'])
@pytest.mark.parametrize('resume', [False, True])
def test_training_updates_sync_resume_and_checkpoint_boundaries(monkeypatch, mode, resume):
    args = build_parser().parse_args([])
    args.decision_only = mode == 'decision'
    args.decision_micro_per_step = 2
    args.decision_loss_weight = 0.5
    args.grad_accum = 2
    args.grad_clip = 100
    args.lr = 0.1
    args.lr_min_ratio = 1
    args.warmup = 0
    args.max_steps = 3
    args.log_every = 1
    args.save_every = 2
    args.resume = 'checkpoint' if resume else None
    args.start_step = 0
    args.wandb = False
    model = ToyModel()
    optim = torch.optim.SGD(model.parameters(), lr=args.lr)
    batch = {k: torch.ones(1, 2, dtype=torch.long) for k in (
        'input_ids', 'labels', 'attention_mask', 'seg_id', 'resp_pos',
        'resp_block', 'turn_id', 'pixel_values', 'image_grid_thw', 'mrope_pos',
    )}
    # Three usable micro-batches per epoch: accumulation must survive the epoch boundary.
    text = [None, batch, batch, batch] if mode != 'decision' else None
    decision = itertools.repeat({'questions': [{'variant': 0}]}) if mode != 'text' else None
    saved, logs = [], []

    def save(model, optimizer, step, args):
        assert model.weight.grad is None  # saves happen after step + zero_grad
        saved.append(step)

    load = Mock(return_value=1)
    monkeypatch.setattr(trainer, 'load_train_state', load)
    monkeypatch.setattr(trainer, 'save_checkpoint', save)
    monkeypatch.setattr(trainer, 'log', logs.append)
    monkeypatch.setattr(trainer.dist, 'all_reduce', lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    trainer.run_training(args, 'cpu', model, TrainingLoaders(text, decision), optim)

    steps = 2 if resume else 3
    # text: grad=2*w; mixed: grad=2*w + 0.5*4*w; Decision-only: grad=0.5*4*w.
    factor = 0.6 if mode == 'mixed' else 0.8
    assert model.weight.item() == pytest.approx(factor ** steps, abs=1e-6)
    assert saved == [2, 3]
    assert len(logs) == steps
    assert [e[1] for e in model.events if e[0] == 'sync'] == {
        'text': [False, True] * steps,
        'mixed': [False, False, False, True] * steps,
        'decision': [False, True] * steps,
    }[mode]
    assert sum(e[0] == 'text' for e in model.events) == (0 if mode == 'decision' else 2 * steps)
    assert sum(e[0] == 'decision' for e in model.events) == (0 if mode == 'text' else 2 * steps)
    for kind, kwargs in model.events:
        if kind == 'text':
            assert set(kwargs) == {'seg_id', 'resp_pos', 'resp_block', 'turn_id',
                                   'pixel_values', 'image_grid_thw', 'mrope_pos'}
    assert load.call_count == int(resume)


def test_optimizer_preserves_parameter_group_scales():
    args = build_parser().parse_args([])
    base = torch.nn.Parameter(torch.ones(1))
    head = torch.nn.Parameter(torch.ones(1))
    model = SimpleNamespace(is_vl=False, param_groups=Mock(return_value=[
        {'params': [base]}, {'params': [head], 'lr_scale': 5},
    ]))
    optimizer = build_optimizer(args, model)
    model.param_groups.assert_called_once_with(
        projector_lr_scale=args.projector_lr_scale, decision_head_lr_scale=args.decision_head_lr_scale,
    )
    assert optimizer.defaults['betas'] == (0.9, args.adam_beta2)
    assert optimizer.defaults['weight_decay'] == args.weight_decay
    assert optimizer.param_groups[1]['lr_scale'] == 5
    assert get_lr(0, 1, 2, 10, 0.1) == 0.5
    assert get_lr(10, 1, 2, 10, 0.1) == 0.1


def test_checkpoint_saves_weights_before_optimizer(monkeypatch):
    events = []
    monkeypatch.setattr(checkpoint, 'save_hf', lambda *a: events.append(('weights', a[-1])))
    monkeypatch.setattr(checkpoint, 'save_train_state', lambda *a: events.append(('optimizer', a[-1])))
    checkpoint.save_checkpoint(None, None, 12, None)
    assert events == [('weights', 'step_12'), ('optimizer', 'step_12')]


def test_distributed_binds_gpu_before_initializing_process_group(monkeypatch):
    events = []
    monkeypatch.setenv('LOCAL_RANK', '2')
    monkeypatch.setattr(torch.cuda, 'set_device', lambda n: events.append(('device', n)))
    monkeypatch.setattr(distributed.dist, 'init_process_group',
                        lambda backend, device_id: events.append((backend, str(device_id))))
    monkeypatch.setattr(torch, 'manual_seed', lambda seed: events.append(('seed', seed)))
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    assert distributed.setup_distributed(SimpleNamespace(seed=42)) == 2
    assert events == [('device', 2), ('nccl', 'cuda:2'), ('seed', 42)]


def test_legacy_model_imports_share_the_implementation():
    import train.hf_block_diffusion as old_dense
    import train.hf_block_diffusion_hybrid as old_hybrid
    from train.models import block_diffusion, hybrid
    assert old_dense is block_diffusion
    assert old_hybrid is hybrid
    assert old_hybrid._cu_seqlens_from_seg is hybrid._cu_seqlens_from_seg


@pytest.mark.parametrize('pack,multiturn,collator', [
    (False, False, 'TextSftCollator'),
    (False, True, 'MultiTurnTextSftCollator'),
    (True, False, 'StreamPackedCollator'),
    (True, True, 'StreamPackedMultiTurnCollator'),
])
def test_data_loader_preserves_text_paths(monkeypatch, pack, multiturn, collator):
    from train.data import loaders
    args = build_parser().parse_args([])
    args.dataset = 'local-data'
    args.pack, args.multiturn = pack, multiturn
    args.linear_block_mode = 'bidirectional'
    monkeypatch.setattr(loaders, 'download_jsonl_paths', Mock(return_value=['data.jsonl']))
    barrier = Mock()
    monkeypatch.setattr(loaders.dist, 'barrier', barrier)
    monkeypatch.setattr(loaders, 'is_master', lambda: True)
    monkeypatch.setattr(loaders, 'rank', lambda: 0)
    monkeypatch.setattr(loaders, 'world_size', lambda: 1)
    for name in ['JsonlAgenticDataset', 'PackedRowDataset', 'PackedMultiTurnRowDataset',
                 'TextSftCollator', 'MultiTurnTextSftCollator',
                 'StreamPackedCollator', 'StreamPackedMultiTurnCollator', 'DataLoader']:
        monkeypatch.setattr(loaders, name, Mock())
    model = SimpleNamespace(tokenizer=object())
    result = loaders.build_data_loaders(args, model)
    assert result.decision is None
    assert result.text is loaders.DataLoader.return_value
    assert loaders.DataLoader.call_args.kwargs['collate_fn'] is getattr(loaders, collator).return_value
    assert loaders.DataLoader.call_args.kwargs['batch_size'] == args.micro_batch_size
    assert loaders.DataLoader.call_args.kwargs['drop_last'] is True
    assert loaders.PackedRowDataset.call_count == int(pack and not multiturn)
    assert loaders.PackedMultiTurnRowDataset.call_count == int(pack and multiturn)
    barrier.assert_called_once()


def test_sharding_orders_children_before_root_and_warms_optional_units(monkeypatch):
    class Unit(torch.nn.Module):
        def unshard(self):
            events.append(('unshard', self))

        def reshard(self):
            events.append(('reshard', self))

    model = Unit()
    model.tm = torch.nn.Module()
    model.tm.layers = torch.nn.ModuleList([Unit(), Unit()])
    model.is_vl = True
    model.visual = torch.nn.Module()
    model.visual.blocks = torch.nn.ModuleList([Unit()])
    model.visual.merger = Unit()
    model.decision_head = Unit()
    args = SimpleNamespace(fsdp_keep_params=True, model_id='toy', fp32_master=False)
    events = []
    monkeypatch.setattr(distributed, 'fully_shard',
                        lambda unit, **kwargs: events.append(('shard', unit, kwargs)))
    monkeypatch.setattr(distributed, 'FSDPModule', Unit)
    distributed.shard_model(model, args)
    children = [*model.tm.layers, *model.visual.blocks, model.visual.merger, model.decision_head]
    sharded = [e for e in events if e[0] == 'shard']
    assert [e[1] for e in sharded] == children + [model]
    for event in sharded[:3]:
        assert event[2]['reshard_after_forward'] is False
    assert [e[:2] for e in events if e[0] != 'shard'] == [
        (operation, child) for child in children for operation in ('unshard', 'reshard')
    ]
    assert model.training


@pytest.mark.parametrize('hybrid', [False, True])
def test_build_model_selects_implementation_and_resume_source(monkeypatch, tmp_path, hybrid):
    from train.models import build
    args = build_parser().parse_args([])
    args.resume = str(tmp_path)
    args.fp32_master = True
    args.decision_data = 'records.jsonl' if hybrid else ''
    model = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
    model.is_vl = True
    model.decision_head = torch.nn.Linear(1, 1) if hybrid else None
    if hybrid:
        saved_head = {key: torch.full_like(value, 0.25) for key, value in model.decision_head.state_dict().items()}
        torch.save({'head_dim': args.decision_head_dim, 'head': saved_head},
                   tmp_path / 'jev_head.pt')
    dense, hybrid_ctor = Mock(return_value=model), Mock(return_value=model)
    monkeypatch.setattr(build, 'HFBlockDiffusion', dense)
    monkeypatch.setattr(build, 'HFBlockDiffusionHybrid', hybrid_ctor)
    monkeypatch.setattr(build, 'is_hybrid_model_id', Mock(return_value=hybrid))
    image_proc = Mock(return_value=object())
    monkeypatch.setattr(build, 'load_image_processor', image_proc)
    assert build.build_model(args) is model
    constructor = hybrid_ctor if hybrid else dense
    assert constructor.call_args.kwargs['model_id'] == str(tmp_path)
    assert constructor.call_args.kwargs['response_buckets'] == (512, 1024, 2048, 4096)
    assert constructor.call_args.kwargs['dtype'] == torch.bfloat16
    if hybrid:
        assert constructor.call_args.kwargs['decision_head_dim'] == args.decision_head_dim
        assert all(torch.equal(value, saved_head[key]) for key, value in model.decision_head.state_dict().items())
    image_proc.assert_called_once_with(str(tmp_path))
    assert model.image_processor is image_proc.return_value
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.parametrize('suffix,value,expected', [
    ('data', 'records.jsonl', 'records.jsonl'), ('only', None, True),
    ('loss_weight', '0.5', 0.5), ('micro_per_step', '2', 2),
    ('max_tokens', '1024', 1024), ('max_variant_tokens', '2048', 2048),
    ('max_state', '384', 384), ('p_none_pair', '0.1', 0.1),
    ('head_dim', '32', 32), ('head_lr_scale', '5', 5.0),
])
def test_decision_cli_accepts_new_flags_and_rejects_legacy(suffix, value, expected):
    parser = build_parser()
    extra = [] if value is None else [value]
    args = parser.parse_args([f'--decision_{suffix}', *extra])
    assert getattr(args, f'decision_{suffix}') == expected
    assert not hasattr(args, f'jev_{suffix}')
    with pytest.raises(SystemExit) as error:
        parser.parse_args([f'--jev_{suffix}', *extra])
    assert error.value.code == 2


def test_actual_legacy_imports_remain_until_020():
    import importlib
    from train.data import decision as data
    from train.models import decision as model
    with pytest.warns(DeprecationWarning, match='0.2.0'):
        legacy = importlib.import_module('train.jev')
    assert legacy.JevRowDataset is data.DecisionRowDataset
    assert legacy.pack_jev is data.pack_decision
    assert legacy.load_jev_records is data.load_decision_records
    assert legacy.jev_loss is model.decision_loss
    assert legacy.PointerHead is model.PointerHead
    assert importlib.import_module('train.hf_block_diffusion') is importlib.import_module('train.models.block_diffusion')
    assert importlib.import_module('train.hf_block_diffusion_hybrid') is importlib.import_module('train.models.hybrid')


def _decision_test_model():
    from train.models.decision import PointerHead
    from train.models.hybrid import HFBlockDiffusionHybrid

    model = object.__new__(HFBlockDiffusionHybrid)
    torch.nn.Module.__init__(model)
    # Registered under its existing name: no renamed/duplicated state-dict keys.
    model.jev_head = PointerHead(4, 2)
    decoder = SimpleNamespace(embed_tokens=torch.nn.Embedding(10, 4),
                              norm=torch.nn.Identity(), rotary_emb=lambda *a, **kw: None)
    model.model = SimpleNamespace(get_decoder=lambda: decoder)
    model.compute_dtype, model.is_vl, model.grad_checkpoint = torch.float32, False, False
    model._ar_doc_mask = lambda *a, **kw: (None, 'eager')
    model._set_attn_impl = lambda *a: None
    model._run_decoder_stack_ar = lambda hidden, *a, **kw: hidden
    return model


def _baseline():
    def unpack(value):
        if isinstance(value, dict):
            if set(value) == {'tensor', 'dtype'}:
                return torch.tensor(value['tensor'], dtype=getattr(torch, value['dtype'].split('.')[-1]))
            return {key: unpack(item) for key, item in value.items()}
        if isinstance(value, list):
            return [unpack(item) for item in value]
        return value
    return unpack(json.loads((Path(__file__).parent / 'fixtures/decision_138239d.json').read_text()))


def _model_from_baseline(data):
    model = _decision_test_model()
    model.load_state_dict(data['head'], strict=True)
    with torch.no_grad():
        model.tm.embed_tokens.weight.copy_(data['embedding'])
    return model


def test_decision_loss_and_gradients_match_pre_refactor_138239d():
    data = _baseline()
    model = _model_from_baseline(data)
    loss, logs = model(None, None, None, decision=data['batch'])
    # Independently generated with the original forward_jev / jev_loss at 138239d.
    assert loss.item() == pytest.approx(0.7800275087356567, abs=1e-7)
    loss.backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, data['gradients'][name], rtol=1e-6, atol=1e-7)
    assert set(logs) == {'decision_loss', 'decision_acc', 'decision_questions', 'decision_tokens'}


def test_checkpoint_layout_and_adamw_resume_match_pre_refactor_138239d():
    data = _baseline()
    model = _model_from_baseline(data)
    assert model.decision_head is model.jev_head
    assert list(model.state_dict()) == ['jev_head.q.weight', 'jev_head.q.bias', 'jev_head.k.weight', 'jev_head.k.bias']
    groups = model.param_groups(decision_head_lr_scale=5)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    # Literal historical ordering also catches q/k swaps, despite their identical shapes.
    assert [(g['lr_scale'], [names[id(p)] for p in g['params']]) for g in groups] == [
        (1.0, []),
        (5, ['jev_head.q.weight', 'jev_head.q.bias', 'jev_head.k.weight', 'jev_head.k.bias']),
    ]
    assert [(g['lr_scale'], [tuple(p.shape) for p in g['params']]) for g in groups] == [
        (1.0, []), (5, [(2, 4), (2,), (2, 4), (2,)])]
    optimizer = torch.optim.AdamW(groups, lr=0.001)
    # Isolate optimizer compatibility from backend rounding in backward. The loss
    # test above checks computed gradients; AdamW can amplify tiny differences in
    # near-zero gradients (notably k.bias) into different first-step updates.
    for name, parameter in model.named_parameters():
        parameter.grad = data['gradients'][name].clone()
    optimizer.step()
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, data['updated_head'][name])
    historical_state = data['optimizer']
    historical_state['state'] = {int(key): value for key, value in historical_state['state'].items()}
    optimizer.load_state_dict(historical_state)
    head_names = data['groups'][1]['parameters']
    parameters = dict(model.named_parameters())
    for index, name in enumerate(head_names):
        for key, value in historical_state['state'][index].items():
            torch.testing.assert_close(optimizer.state[parameters[name]][key], value)


def test_removed_model_aliases_are_not_accepted():
    model = _decision_test_model()
    with pytest.raises(TypeError, match='jev'):
        model(None, None, None, jev={})
    with pytest.raises(TypeError, match='jev_head_lr_scale'):
        model.param_groups(jev_head_lr_scale=5)
