#!/usr/bin/env bash
# Скачивает модели и датасеты, которые нужны тестам и сэмплам, и раскладывает их по местам.
# В git они не хранятся из-за размера (mnist.csv - 105 МБ) и лежат файлами GitHub-релиза.
# Каждый файл скачивается один раз в build/assets, проверяется по SHA-256 и копируется во все
# каталоги, где его ждут; повторный запуск ничего не скачивает.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/download-assets.sh [asset...]

Downloads the models and datasets used by the tests and samples and copies them where they are
expected. Without arguments every asset is downloaded.

Assets:
  mnist.tflite              tests
  mnist.csv                 tests
  chess-ai.tflite           compose-resources and moko-resources samples
  mobile_net_ssd_v2.tflite  vision sample

Set KTENSORFLOW_ASSETS_URL to download from another location.
EOF
}

RELEASE_URL="${KTENSORFLOW_ASSETS_URL:-https://github.com/kursor1337/KTensorFlow/releases/download/assets}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/build/assets"

# имя|SHA-256|каталоги назначения относительно корня репозитория
ASSETS="
mnist.tflite|66742159ff92a669b49fedeec92f9280a5598392bf70f1f5826573be8ade5824|ktensorflow-test/src/androidDeviceTest/resources ktensorflow-test/src/iosTest/resources
mnist.csv|06e64b0bf65de89c522348319ce5f63f43fb1cb40e1ce3c2ff43c1d3b8a0e14c|ktensorflow-test/src/androidDeviceTest/resources ktensorflow-test/src/iosTest/resources
chess-ai.tflite|b6695d57a8ec81425867448db013e3f42bd35699cfc25d9f3006e95e9497949a|samples/compose-resources/composeApp/src/commonMain/composeResources/files samples/moko-resources/composeApp/src/commonMain/moko-resources/files
mobile_net_ssd_v2.tflite|7227d6cb19ed832e1febdf00851d95cf1ade00ce663bc6ac017b99cc7ee66bb9|samples/vision/composeApp/src/commonMain/composeResources/files
"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

install_asset() {
    local name="$1" checksum="$2" destinations="$3"
    local cached="$CACHE/$name"

    if [ ! -f "$cached" ] || [ "$(sha256 "$cached")" != "$checksum" ]; then
        echo "Downloading $name"
        if ! curl -fL --retry 3 --progress-bar -o "$cached.part" "$RELEASE_URL/$name"; then
            rm -f "$cached.part"
            echo "error: could not download $name from $RELEASE_URL" >&2
            exit 1
        fi
        # Битый или подменённый файл не должен попасть в тесты: они бы падали непонятно почему
        local actual
        actual="$(sha256 "$cached.part")"
        if [ "$actual" != "$checksum" ]; then
            rm -f "$cached.part"
            echo "error: checksum mismatch for $name: expected $checksum, got $actual" >&2
            exit 1
        fi
        mv "$cached.part" "$cached"
    fi

    local destination
    for destination in $destinations; do
        mkdir -p "$ROOT/$destination"
        if ! cmp -s "$cached" "$ROOT/$destination/$name"; then
            cp "$cached" "$ROOT/$destination/$name"
            echo "  $name -> $destination"
        fi
    done
}

main() {
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
        usage
        exit 0
    fi

    local known=""
    local line
    while IFS= read -r line; do
        [ -n "$line" ] && known="$known ${line%%|*}"
    done <<< "$ASSETS"

    local requested
    for requested in "$@"; do
        case " $known " in
            *" $requested "*) ;;
            *)
                echo "error: unknown asset '$requested'" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    mkdir -p "$CACHE"
    local name checksum destinations
    while IFS='|' read -r name checksum destinations; do
        [ -z "$name" ] && continue
        if [ "$#" -gt 0 ]; then
            case " $* " in
                *" $name "*) ;;
                *) continue ;;
            esac
        fi
        install_asset "$name" "$checksum" "$destinations"
    done <<< "$ASSETS"
}

main "$@"
