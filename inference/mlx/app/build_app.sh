#!/bin/zsh
# Builds "Zoro-1 Chat.app" from this Swift package.
#
#   zsh build_app.sh                  dev build: the app runs the server from this trida-stack
#                                     checkout (../.venv after `uv sync`)
#   zsh build_app.sh --bundle         self-contained build: Python 3.12 + trida_mlx + MLX and the
#                                     other dependencies inside the app; nothing else to install
#   zsh build_app.sh --bundle --zip   ...and dist/Zoro-1-Chat-<version>.zip to send to someone
#   zsh build_app.sh --bundle --dmg   ...and dist/Zoro-1-Chat-<version>.dmg (drag-to-Applications disk image)
#   --install                         also copy the app to ~/Applications (or /Applications)
#
# Sharing without Gatekeeper warnings needs a Developer ID certificate and notarization:
#   SIGN_IDENTITY="Developer ID Application: <Team> (<TEAMID>)" NOTARY_PROFILE=<profile> \
#     zsh build_app.sh --bundle --zip
# (create the profile once with: xcrun notarytool store-credentials <profile> --apple-id ... --team-id ...)
set -euo pipefail
setopt null_glob
cd "$(dirname "$0")"
HERE="$PWD"
MLX_DIR="$(cd .. && pwd)"
VERSION=0.3.0
PYVER=3.12

BUNDLE=0; ZIP=0; DMG=0; INSTALL=0
for a in "$@"; do
  case "$a" in
    --bundle) BUNDLE=1 ;;
    --zip) ZIP=1 ;;
    --dmg) DMG=1 ;;
    --install) INSTALL=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

if ! command -v swift >/dev/null; then
  echo "swift not found: install Xcode or the Command Line Tools (xcode-select --install)" >&2
  exit 1
fi

echo "==> building the app"
swift build -c release --arch arm64
BIN="$(swift build -c release --arch arm64 --show-bin-path)/TridaChat"

APP="build/Zoro-1 Chat.app"
# chmod first: copied runtimes can contain read-only directories that rm -rf cannot empty
[[ -e "$APP" ]] && chmod -R u+w "$APP" && rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/TridaChat"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Zoro-1 Chat</string>
  <key>CFBundleDisplayName</key><string>Zoro-1 Chat</string>
  <key>CFBundleIdentifier</key><string>co.trillionlabs.zoro1chat</string>
  <key>CFBundleExecutable</key><string>TridaChat</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSAppTransportSecurity</key>
  <dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict>
</plist>
PLIST

if (( BUNDLE )); then
  command -v uv >/dev/null || { echo "--bundle needs uv (https://docs.astral.sh/uv/)" >&2; exit 1; }
  PYDIR="$APP/Contents/Resources/python"
  PY="$PYDIR/bin/python3"

  echo "==> bundling a relocatable CPython $PYVER (python-build-standalone, via uv)"
  # Ignore any active conda / virtualenv: only uv's own standalone CPython is relocatable.
  unset CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV PYTHONHOME PYTHONPATH
  export UV_PYTHON_PREFERENCE=only-managed
  uv python install "$PYVER"
  UV_PY_DIR="$(uv python dir)"
  PREFIX=""
  for d in "$UV_PY_DIR"/cpython-"$PYVER".*-macos-aarch64-none; do
    [[ -x "$d/bin/python$PYVER" ]] && PREFIX="$d"   # glob is sorted, so the newest patch wins
  done
  if [[ -z "$PREFIX" || -e "$PREFIX/conda-meta" ]]; then
    echo "could not find uv's standalone CPython $PYVER under $UV_PY_DIR" >&2
    exit 1
  fi
  echo "    using $PREFIX"
  cp -R "$PREFIX" "$PYDIR"
  find "$PYDIR/lib" -maxdepth 2 -name EXTERNALLY-MANAGED -delete

  echo "==> installing trida_mlx and its locked dependencies"
  REQ="$(mktemp -t zoro-requirements)"
  (cd "$MLX_DIR" && uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt -o "$REQ" >/dev/null)
  # uv.lock also resolves Linux (CI) packages; drop the Linux-only MLX backends even if an older uv
  # wrote them without their platform marker (they have no macOS wheels).
  grep -vE '^(mlx-cpu|mlx-cuda[-0-9]*)==' "$REQ" > "$REQ.mac" && mv "$REQ.mac" "$REQ"
  "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --default-pip >/dev/null
  PIP=("$PY" -m pip install -q --progress-bar off --no-cache-dir --disable-pip-version-check --no-warn-script-location)
  "${PIP[@]}" -r "$REQ"
  "${PIP[@]}" --no-deps "$MLX_DIR"
  rm -f "$REQ"

  echo "==> trimming"
  "$PY" -m pip uninstall -y -q pip >/dev/null 2>&1 || true
  rm -rf "$PYDIR/include" "$PYDIR/share"
  rm -rf "$PYDIR/lib/python$PYVER/"{test,idlelib,tkinter,turtledemo,ensurepip,lib2to3} 2>/dev/null || true
  rm -rf "$PYDIR/lib/"tcl* "$PYDIR/lib/"tk* "$PYDIR/lib/"itcl* "$PYDIR/lib/"libtcl* "$PYDIR/lib/"libtk* 2>/dev/null || true
  # console scripts carry absolute shebangs of this build machine; the app uses `python3 -m` instead
  find "$PYDIR/bin" -type f ! -name 'python3*' -delete
  find "$PYDIR" -name __pycache__ -type d -prune -exec rm -rf {} +
  # precompile now: at run time the interpreter must not write into the signed bundle
  "$PY" -m compileall -q -j 0 "$PYDIR/lib" >/dev/null || true

  echo "==> smoke test"
  env -i HOME="$HOME" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$PY" -c \
    'import mlx.core as mx, mlx_lm, transformers, huggingface_hub, PIL, trida_mlx.server, trida_mlx.convert; print("    bundled runtime OK: mlx", mx.__version__, "on", mx.default_device())'
  SIZE_MB=$(du -sm "$PYDIR" | cut -f1)
  echo "    runtime size: ${SIZE_MB} MB"
  if (( SIZE_MB > 1200 )); then
    echo "runtime is ${SIZE_MB} MB, far above the expected ~500 MB; check what was copied into $PYDIR" >&2
    exit 1
  fi
fi

if [[ -n "${SIGN_IDENTITY:-}" ]]; then
  echo "==> signing with $SIGN_IDENTITY (hardened runtime)"
  ENT="$HERE/entitlements.plist"
  if [[ -d "$APP/Contents/Resources/python" ]]; then
    # every Mach-O inside the runtime: extension modules, dylibs, the interpreter itself
    find "$APP/Contents/Resources/python" -type f \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \) -print0 |
      while IFS= read -r -d '' f; do
        if file -b "$f" | grep -q 'Mach-O'; then
          codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$SIGN_IDENTITY" "$f"
        fi
      done
  fi
  codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$SIGN_IDENTITY" "$APP"
  codesign --verify --deep --strict "$APP"
else
  # Ad-hoc signature: runs on this Mac; on others macOS asks the user to allow it (see README).
  codesign --force --deep --sign - "$APP"
fi
echo "built: $HERE/$APP ($(du -sh "$APP" | cut -f1))"

if (( ZIP )); then
  mkdir -p dist
  OUT="dist/Zoro-1-Chat-$VERSION.zip"
  rm -f "$OUT"
  ditto -c -k --keepParent "$APP" "$OUT"
  if [[ -n "${SIGN_IDENTITY:-}" && -n "${NOTARY_PROFILE:-}" ]]; then
    echo "==> notarizing (a few minutes)"
    xcrun notarytool submit "$OUT" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$APP"
    rm -f "$OUT"
    ditto -c -k --keepParent "$APP" "$OUT"
  fi
  echo "zip: $HERE/$OUT ($(du -sh "$OUT" | cut -f1))"
fi

if (( DMG )); then
  mkdir -p dist
  OUT="dist/Zoro-1-Chat-$VERSION.dmg"
  STAGE="$(mktemp -d -t zoro-dmg)"
  cp -R "$APP" "$STAGE/"
  ln -s /Applications "$STAGE/Applications"
  cat > "$STAGE/If it says damaged.txt" <<'TXT'
Zoro-1 Chat

1. Drag "Zoro-1 Chat" onto the Applications folder.

2. If macOS says the app "is damaged and can't be opened": the app is fine. macOS marks
   every app that comes from another computer and is not notarized by Apple. Remove the
   mark once by opening Terminal and running:

       xattr -dr com.apple.quarantine "/Applications/Zoro-1 Chat.app"

   then open the app again.

3. On first launch, use "Get a Model" to download Zoro-1 (needs a Hugging Face account that
   has accepted the model's terms, and about 15 GB of free disk space).

Requires an Apple Silicon Mac.
TXT
  rm -f "$OUT"
  hdiutil create -volname "Zoro-1 Chat" -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov "$OUT" >/dev/null
  rm -rf "$STAGE"
  if [[ -n "${SIGN_IDENTITY:-}" ]]; then
    codesign --force --timestamp --sign "$SIGN_IDENTITY" "$OUT"
    if [[ -n "${NOTARY_PROFILE:-}" ]]; then
      echo "==> notarizing the disk image (a few minutes)"
      xcrun notarytool submit "$OUT" --keychain-profile "$NOTARY_PROFILE" --wait
      xcrun stapler staple "$OUT"
    fi
  fi
  echo "dmg: $HERE/$OUT ($(du -sh "$OUT" | cut -f1))"
fi

if (( INSTALL )); then
  DEST=~/Applications
  mkdir -p "$DEST" 2>/dev/null || true
  if [[ ! -w "$DEST" ]]; then DEST=/Applications; fi
  [[ -e "$DEST/Zoro-1 Chat.app" ]] && chmod -R u+w "$DEST/Zoro-1 Chat.app" 2>/dev/null
  if rm -rf "$DEST/Zoro-1 Chat.app" 2>/dev/null && cp -R "$APP" "$DEST/" 2>/dev/null; then
    echo "installed: $DEST/Zoro-1 Chat.app"
  else
    echo "could not write to $DEST; run it from here:  open \"$HERE/$APP\"" >&2
  fi
fi
