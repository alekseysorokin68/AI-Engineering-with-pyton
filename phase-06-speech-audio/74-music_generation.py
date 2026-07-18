"""
Музыкальная генерация на Python
Основы: марковские цепи, пентатоника, аккорды, визуализация нот
"""

import random

random.seed(42)

# === Музыкальные константы ===

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Маппинг нот в MIDI-номера (октава 4)
NOTE_TO_MIDI = {name: 60 + i for i, name in enumerate(NOTE_NAMES)}

# Пентатоника мажор (ноты относительно тоники)
MAJOR_PENTATONIC_INTERVALS = [0, 2, 4, 7, 9]

# Пентатоника минор (ноты относительно тоники)
MINOR_PENTATONIC_INTERVALS = [0, 2, 3, 7, 9]

# Интервалы для аккордов
CHORD_INTERVALS = {
    'major': [0, 4, 7],
    'minor': [0, 3, 7],
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'dom7': [0, 4, 7, 10],
}

# Ноты для марковской цепи (мажорная гамма)
MAJOR_SCALE = ['C', 'D', 'E', 'F', 'G', 'A', 'B']


def midi_to_name(midi_note):
    """MIDI-номер -> имя ноты с октавой"""
    octave = (midi_note // 12) - 1
    name = NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"


def get_scale_notes(root, scale_type='major', octave=4):
    """Получить ноты гаммы"""
    root_midi = NOTE_TO_MIDI[root] + (octave - 4) * 12
    if scale_type == 'major':
        intervals = [0, 2, 4, 5, 7, 9, 11]
    elif scale_type == 'minor':
        intervals = [0, 2, 3, 5, 7, 8, 10]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]
    return [midi_to_name(root_midi + i) for i in intervals]


def get_pentatonic_notes(root, scale_type='major', octave=4):
    """Получить пентатонические ноты"""
    root_midi = NOTE_TO_MIDI[root] + (octave - 4) * 12
    intervals = MAJOR_PENTATONIC_INTERVALS if scale_type == 'major' else MINOR_PENTATONIC_INTERVALS
    return [midi_to_name(root_midi + i) for i in intervals]


# === Марковские цепи ===

class MarkovChain:
    """Марковская цепь первого порядка для генерации мелодий"""

    def __init__(self):
        self.transitions = {}

    def train(self, sequence):
        """Обучить цепь на последовательности нот"""
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_note = sequence[i + 1]
            if current not in self.transitions:
                self.transitions[current] = {}
            self.transitions[current][next_note] = \
                self.transitions[current].get(next_note, 0) + 1

    def generate(self, length=16, start=None):
        """Сгенерировать последовательность"""
        if not self.transitions:
            raise ValueError("Цепь не обучена")

        if start is None:
            start = random.choice(list(self.transitions.keys()))

        result = [start]
        for _ in range(length - 1):
            current = result[-1]
            if current not in self.transitions:
                break
            weights = self.transitions[current]
            notes = list(weights.keys())
            probs = list(weights.values())
            result.append(random.choices(notes, weights=probs, k=1)[0])
        return result


# === Визуализация ===

def visualize_notes(notes, width=60):
    """Визуализация нот как ASCII-диаграмма"""
    if not notes:
        return ""

    # Определяем диапазон
    all_notes_flat = []
    for note in notes:
        if isinstance(note, str):
            name = note[:-1] if note[-1].isdigit() else note
            if name in NOTE_TO_MIDI:
                all_notes_flat.append(NOTE_TO_MIDI[name])
            else:
                all_notes_flat.append(60)

    if not all_notes_flat:
        return ""

    min_note = min(all_notes_flat)
    max_note = max(all_notes_flat)
    note_range = max(max_note - min_note + 1, 1)

    # Ограничиваем высоту
    display_height = min(note_range, 20)
    step = max(note_range // display_height, 1)

    lines = []
    for row in range(display_height, -1, -1):
        midi_val = min_note + row * step
        note_label = midi_to_name(midi_val).ljust(4)
        line = note_label + "|"
        for note in notes:
            if isinstance(note, str):
                name = note[:-1] if note[-1].isdigit() else note
                if name in NOTE_TO_MIDI:
                    val = NOTE_TO_MIDI[name]
                else:
                    val = 60
            else:
                val = note
            if abs(val - midi_val) < step:
                line += " * "
            else:
                line += "   "
        lines.append(line)

    return "\n".join(lines)


def print_chord(notes, label=""):
    """Красивый вывод аккорда"""
    midi_vals = []
    for n in notes:
        name = n[:-1] if n[-1].isdigit() else n
        if name in NOTE_TO_MIDI:
            midi_vals.append(NOTE_TO_MIDI[name])

    intervals_str = []
    for i in range(1, len(midi_vals)):
        intervals_str.append(str(midi_vals[i] - midi_vals[0]))

    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}Ноты: {' → '.join(notes)}  |  Интервалы: {', '.join(intervals_str)}")


# === Демо 1: Марковская цепь для мелодий ===

def demo1_markov_melody():
    print("=" * 60)
    print("  ДЕМО 1: МАРКОВСКАЯ ЦЕПЬ ДЛЯ МЕЛОДИЙ")
    print("=" * 60)

    # Обучающая мелодия (фрагмент в стиле блюза)
    training_melody = [
        'C4', 'E4', 'G4', 'A4', 'G4', 'E4', 'C4', 'D4',
        'E4', 'F4', 'E4', 'D4', 'C4', 'G3', 'A3', 'B3',
        'C4', 'D4', 'E4', 'G4', 'A4', 'G4', 'F4', 'E4',
        'D4', 'C4', 'E4', 'G4', 'C5', 'B4', 'A4', 'G4',
    ]

    print("\nОбучающая мелодия:")
    print(f"  {' → '.join(training_melody)}")

    # Обучаем марковскую цепь
    mc = MarkovChain()
    mc.train(training_melody)

    print("\nМатрица переходов (топ-3 для каждой ноты):")
    for note in sorted(mc.transitions.keys()):
        trans = mc.transitions[note]
        top3 = sorted(trans.items(), key=lambda x: -x[1])[:3]
        trans_str = ", ".join(f"{n}:{c}" for n, c in top3)
        print(f"  {note} → [{trans_str}]")

    # Генерируем новую мелодию
    generated = mc.generate(length=16, start='C4')
    print(f"\nСгенерированная мелодия:")
    print(f"  {' → '.join(generated)}")

    print("\nВизуализация:")
    print(visualize_notes(generated))
    print()


# === Демо 2: Пентатоника ===

def demo2_pentatonic():
    print("=" * 60)
    print("  ДЕМО 2: ПЕНТАТОНИКА")
    print("=" * 60)

    # Мажорная пентатоника
    print("\n--- Мажорная пентатоника ---")
    for root in ['C', 'G', 'D', 'A']:
        notes = get_pentatonic_notes(root, 'major')
        intervals = MAJOR_PENTATONIC_INTERVALS
        print(f"\n  {root} мажорная пентатоника:")
        print(f"    Ноты: {', '.join(notes)}")
        print(f"    Интервалы (полутоны): {intervals}")

    # Минорная пентатоника
    print("\n--- Минорная пентатоника ---")
    for root in ['A', 'E', 'B']:
        notes = get_pentatonic_notes(root, 'minor')
        print(f"\n  {root} минорная пентатоника:")
        print(f"    Ноты: {', '.join(notes)}")

    # Визуализация паттернов пентатоники
    print("\n--- Визуализация паттернов ---")
    for root in ['C', 'G']:
        notes = get_pentatonic_notes(root, 'major')
        midi_vals = []
        for n in notes:
            name = n[:-1] if n[-1].isdigit() else n
            if name in NOTE_TO_MIDI:
                midi_vals.append(NOTE_TO_MIDI[name])

        print(f"\n  {root} major pentatonic:")
        for i, (note, midi) in enumerate(zip(notes, midi_vals)):
            bar_len = (midi - 60) // 2 + 1
            print(f"    {note:>4} {'█' * bar_len} (MIDI {midi})")

    print()


# === Демо 3: Аккорды ===

def demo3_chords():
    print("=" * 60)
    print("  ДЕМО 3: ГЕНЕРАЦИЯ АККОРДОВ")
    print("=" * 60)

    # Основные трезвучия
    print("\n--- Основные трезвучия ---")
    for chord_type in ['major', 'minor', 'dim', 'aug']:
        print(f"\n  {chord_type.upper()} трезвучие:")
        for root in ['C', 'D', 'E', 'F', 'G', 'A']:
            root_midi = NOTE_TO_MIDI[root]
            chord_notes = [midi_to_name(root_midi + i) for i in CHORD_INTERVALS[chord_type]]
            print_chord(chord_notes, root)

    # Септаккорды
    print("\n--- Септаккорды ---")
    for chord_type in ['maj7', 'min7', 'dom7']:
        print(f"\n  {chord_type}:")
        for root in ['C', 'D', 'E', 'F', 'G', 'A']:
            root_midi = NOTE_TO_MIDI[root]
            chord_notes = [midi_to_name(root_midi + i) for i in CHORD_INTERVALS[chord_type]]
            print_chord(chord_notes, root)

    # Прогрессии аккордов
    print("\n--- Типовые аккордовые прогрессии ---")

    progressions = {
        'I-IV-V-I (блюз/поп)': ['C', 'F', 'G', 'C'],
        'I-V-vi-IV (поп)': ['C', 'G', 'Am', 'F'],
        'ii-V-I (джаз)': ['Dm', 'G', 'C'],
        'I-vi-IV-V (50-е)': ['C', 'Am', 'F', 'G'],
    }

    for name, prog in progressions.items():
        print(f"\n  {name}:")
        for chord in prog:
            if len(chord) > 1 and chord[1] in ('m', '7'):
                root = chord[0]
                ctype = chord[1:]
                if ctype == 'm':
                    ctype = 'minor'
                elif ctype == '7':
                    ctype = 'dom7'
                root_midi = NOTE_TO_MIDI[root]
                notes = [midi_to_name(root_midi + i) for i in CHORD_INTERVALS[ctype]]
            else:
                root_midi = NOTE_TO_MIDI[chord]
                notes = [midi_to_name(root_midi + i) for i in CHORD_INTERVALS['major']]
            print(f"    {chord:>3} → {', '.join(notes)}")

    print()


# === Демо 4: Генерация мелодии с аккордами ===

def demo4_generate_melody():
    print("=" * 60)
    print("  ДЕМО 4: ГЕНЕРАЦИЯ МЕЛОДИИ")
    print("=" * 60)

    # Генерируем мелодию над аккордовой прогрессией
    root = 'C'
    pentatonic = get_pentatonic_notes(root, 'major')
    print(f"\nПентатоника: {', '.join(pentatonic)}")

    progression = [
        ('C', 'major'),
        ('F', 'major'),
        ('G', 'major'),
        ('C', 'major'),
    ]

    print("Прогрессия: C → F → G → C")

    # Создаём мелодию, привязанную к аккордам
    melody = []
    for chord_root, chord_type in progression:
        chord_midi = NOTE_TO_MIDI[chord_root]
        chord_tones = [chord_midi + i for i in CHORD_INTERVALS[chord_type]]

        # 4 ноты на каждый аккорд
        for _ in range(4):
            if random.random() < 0.6:
                # Выбираем ноту из аккорда
                note_midi = random.choice(chord_tones)
            else:
                # Выбираем из пентатоники (в октаве 4)
                penta_midi = [NOTE_TO_MIDI[n[:-1]] + 12 for n in pentatonic]
                note_midi = random.choice(penta_midi)
            melody.append(midi_to_name(note_midi))

    print(f"\nСгенерированная мелодия:")
    print(f"  {' → '.join(melody)}")

    print("\nВизуализация мелодии:")
    print(visualize_notes(melody))

    # Анализ нот
    print("\n--- Анализ мелодии ---")
    note_counts = {}
    for note in melody:
        name = note[:-1] if note[-1].isdigit() else note
        note_counts[name] = note_counts.get(name, 0) + 1

    print("  Частота нот:")
    for note, count in sorted(note_counts.items(), key=lambda x: -x[1]):
        bar = '█' * (count * 4)
        print(f"    {note:>3}: {bar} ({count})")

    # Интервалы
    print("\n  Интервалы между нотами:")
    for i in range(1, min(len(melody), 16)):
        prev_midi = NOTE_TO_MIDI[melody[i-1][:-1]] if melody[i-1][:-1] in NOTE_TO_MIDI else 60
        curr_midi = NOTE_TO_MIDI[melody[i][:-1]] if melody[i][:-1] in NOTE_TO_MIDI else 60
        interval = curr_midi - prev_midi
        direction = "↑" if interval > 0 else "↓" if interval < 0 else "="
        print(f"    {melody[i-1]} → {melody[i]}: {direction}{abs(interval)} полутонов")

    print()


# === Запуск всех демо ===

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           ОСНОВЫ ГЕНЕРАЦИИ МУЗЫКИ НА PYTHON                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    demo1_markov_melody()
    demo2_pentatonic()
    demo3_chords()
    demo4_generate_melody()

    print("=" * 60)
    print("  ВСЕ ДЕМО ЗАВЕРШЕНЫ")
    print("=" * 60)
