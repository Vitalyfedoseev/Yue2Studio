#!/bin/bash
# Скачивает саундфонты FluidR3_GM (по ноте A0..C8) для ABC-редактора (/abc).
# Папка static/vendor/abcjs/soundfonts/ в git не входит — стягивается этим скриптом.
# Идемпотентен: существующие файлы не перекачивает. Новый инструмент — допишите
# его имя в INSTRUMENTS и запустите снова.
set -u
BASE="https://paulrosen.github.io/midi-js-soundfonts/FluidR3_GM"
DEST="$(cd "$(dirname "$0")/.." && pwd)/static/vendor/abcjs/soundfonts"
INSTRUMENTS="acoustic_grand_piano electric_piano_1 harpsichord glockenspiel vibraphone music_box drawbar_organ church_organ accordion harmonica acoustic_guitar_nylon acoustic_guitar_steel electric_guitar_jazz electric_guitar_clean electric_guitar_muted overdriven_guitar distortion_guitar acoustic_bass electric_bass_finger electric_bass_pick violin viola cello contrabass pizzicato_strings orchestral_harp timpani string_ensemble_1 string_ensemble_2 choir_aahs trumpet muted_trumpet trombone french_horn alto_sax tenor_sax clarinet oboe flute pan_flute"
# Апстрим хранит ноты под бемольными именами (Gb3, Bb0) — диезных файлов нет
NOTES=$(python3 - <<'PY'
names = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
print(" ".join(f"{names[m % 12]}{m // 12 - 1}" for m in range(21, 109)))
PY
)
mkdir -p "$DEST"
for ins in $INSTRUMENTS; do
  mkdir -p "$DEST/$ins-mp3"
  for n in $NOTES; do echo "$ins $n"; done
done | xargs -P 12 -n 2 bash -c '
  ins="$0"; n="$1"
  f="'"$DEST"'/$ins-mp3/$n.mp3"
  [ -s "$f" ] || curl -sf --max-time 40 -o "$f" "'"$BASE"'/$ins-mp3/$n.mp3" || rm -f "$f"
'
# валидация: настоящий mp3 (MPEG-синхроголовка или ID3), а не 404-страница
bad=0
while IFS= read -r -d '' f; do
  if ! head -c 3 "$f" | grep -q $'ID3' && [ "$(( $(od -An -tu1 -N2 "$f" | tr -d " ") / 256 & 0xE0 ))" != 224 ]; then
    rm -f "$f"; bad=$((bad+1))
  fi
done < <(find "$DEST" -name "*.mp3" -print0)
total=$(find "$DEST" -name "*.mp3" | wc -l)
echo "файлов: $total, битых удалено: $bad"
[ "$bad" = 0 ] || echo "часть нот не скачалась — запустите скрипт ещё раз"
