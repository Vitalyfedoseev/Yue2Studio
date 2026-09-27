#!/usr/bin/env python3
"""Прореживание MIDI до монофонической мелодии (для каверов YuE2).

Полномиксовая транскрипция (basic-pitch по целому миксу) даёт тысячи нот
аккордами — в контекст модели (24576 токенов) такая партитура не лезет, а для
cot=melody нужна ведущая линия. На каждом онсете (окно 30 мс) оставляем самую
длинную ноту (устойчивые вокал/лид против коротких транзиентов), при равенстве
— верхнюю по высоте; наложения подрезаются.

Запуск: melody.py <в.mid> <out.mid>; stdout: JSON {notes_in, notes_out}.
"""
import json
import sys
from pathlib import Path

import pretty_midi

ONSET_GROUP_S = 0.03  # ноты, начавшиеся в пределах 30 мс — один «онсет»


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("использование: melody.py <в.mid> <out.mid>")
    pm = pretty_midi.PrettyMIDI(sys.argv[1])

    notes = sorted(
        (n.start, n.end, n.pitch)
        for inst in pm.instruments
        for n in inst.notes
    )

    # группировка по онсетам, в группе — самая длинная нота (тяготеем к лиду)
    kept: list[tuple[float, float, int]] = []
    group: list[tuple[float, float, int]] = []
    for note in notes:
        if group and note[0] - group[0][0] > ONSET_GROUP_S:
            kept.append(max(group, key=lambda n: (n[1] - n[0], n[2])))
            group = []
        group.append(note)
    if group:
        kept.append(max(group, key=lambda n: (n[1] - n[0], n[2])))

    # подрезаем наложения: нота не должна глушить старт следующей
    for i, (s, e, p) in enumerate(kept):
        if i + 1 < len(kept) and e > kept[i + 1][0]:
            kept[i] = (s, max(kept[i + 1][0], s + 0.04), p)

    out = pretty_midi.PrettyMIDI(initial_tempo=pm.get_tempo_changes()[1][0]
                                  if len(pm.get_tempo_changes()[1]) else 120)
    if pm.time_signature_changes:
        out.time_signature_changes.append(pm.time_signature_changes[0])
    inst = pretty_midi.Instrument(program=0)
    inst.notes = [pretty_midi.Note(velocity=90, pitch=p, start=s, end=e)
                  for s, e, p in kept]
    out.instruments.append(inst)
    out.write(sys.argv[2])
    print(json.dumps({"notes_in": len(notes), "notes_out": len(kept)}))


if __name__ == "__main__":
    main()
