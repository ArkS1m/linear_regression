import sys
import time
import os
import msvcrt

class FeatureSelector:
    def __init__(self, features):
        self.features = features
        self.selected = [0, 1]
        self.current = 0

    def display(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "=" * 60)
        print("ВЫБЕРИТЕ ДВА ПРИЗНАКА ДЛЯ SCATTER PLOT")
        print("↑↓ - перемещение | SPACE - выбор/снятие | ENTER - подтвердить")
        print("=" * 60)

        for i, feature in enumerate(self.features):
            color = "\033[93m" if i == self.current else "\033[92m" if i in self.selected else ""
            reset = "\033[0m"

            print(f"{color}{feature}{reset}")

        print("\nВыбрано признаков:", len(self.selected))
        print("=" * 60)

    def run(self):
        print("Настройка клавиш... (нажмите Ctrl+C для выхода)")
        time.sleep(1)

        while True:
            self.display()

            try:
                # Ожидание нажатия клавиши (имитация через input для простоты)
                print("\nНажмите стрелку ↑↓, SPACE или ENTER: ", end="")
                key = msvcrt.getch()
                print(key)

                if key == b'\r':
                    if len(self.selected) == 2:
                        return [self.features[i] for i in self.selected]

                elif key == b' ':
                    if self.current in self.selected:
                        self.selected.remove(self.current)
                    else:
                        if len(self.selected) > 1:
                            self.selected.remove(self.selected[0])

                        self.selected.append(self.current)

                elif key == b'H':
                    self.current = (self.current - 1) % len(self.features)

                elif key == b'P':
                    self.current = (self.current + 1) % len(self.features)

            except KeyboardInterrupt:
                sys.exit(0)