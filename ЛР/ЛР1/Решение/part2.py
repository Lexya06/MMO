"""
Лабораторная работа №1 по курсу "Методы машинного обучения"
Часть 2: Функции в Python, обработка исключений (try-except-finally), работа с матрицами и структурами данных
Вариант: 2
"""

from collections import Counter


# ==============================================================================
# Задание 1 (2 балла):
# Написать функцию для определения анаграмм.
# Два слова являются анаграммами, если состоят из одинаковых букв.
# Например, анаграммами являются слова "binary" и "brainy", или "раздвоение" и "дозревание".
# Требование: использовать функции и блок try/except/finally.
# ==============================================================================
def are_anagrams(word1: str, word2: str) -> bool:
    """
    Проверяет, являются ли два слова word1 и word2 анаграммами.
    Анаграммы содержат одинаковые буквы в одинаковых количествах,
    независимо от регистра и пробелов.
    """
    try:
        # Проверка типов входных аргументов
        if not isinstance(word1, str) or not isinstance(word2, str):
            raise TypeError("Оба аргумента должны быть строками (str)!")

        # Приводим к нижнему регистру и удаляем пробелы/знаки
        cleaned_w1 = "".join(ch.lower() for ch in word1 if ch.isalnum())
        cleaned_w2 = "".join(ch.lower() for ch in word2 if ch.isalnum())

        if not cleaned_w1 or not cleaned_w2:
            raise ValueError(
                "Слова не должны быть пустыми после очистки от пробелов!")

        # Сравниваем отсортированные списки букв
        result = sorted(cleaned_w1) == sorted(cleaned_w2)
        return result

    except TypeError as te:
        print(f"[Ошибка типа]: {te}")
        return False
    except ValueError as ve:
        print(f"[Ошибка значения]: {ve}")
        return False
    except Exception as e:
        print(f"[Непредвиденная ошибка]: {e}")
        return False
    finally:
        # Блок finally выполняется всегда, независимо от того, было ли
        # исключение
        print("  -> Завершение вызова функции are_anagrams (блок finally выполнен)")


def run_task1():
    print("=" * 60)
    print("ЗАДАНИЕ 1: Определение анаграмм")
    print("=" * 60)

    test_pairs = [
        ("binary", "brainy"),
        ("раздвоение", "дозревание"),
        ("Апельсин", "Спаниель"),
        ("кот", "кит"),
        ("Hello", "World"),
        (12345, "54321")  # Тест на обработку исключения
    ]

    for w1, w2 in test_pairs:
        try:
            res = are_anagrams(w1, w2)
            print(f"Слова: '{w1}' и '{w2}' -> Являются анаграммами: {res}")
        except Exception as err:
            print(f"Ошибка при обработке пары ('{w1}', '{w2}'): {err}")
        finally:
            print("-" * 40)
    print()


# ==============================================================================
# Задание 2 (4 балла):
# Напишите функцию, которая будет принимать один аргумент.
# - Если в функцию передаётся кортеж (tuple), то посчитать его количество элементов.
# - Если список (list), то найти сумму до первого нулевого элемента.
# - Число (int / float) – вывести его в обратном порядке.
# - Строка (str) – кол-во слов. Определить, какой символ в строке встречается чаще всего.
# Сделать проверку со всеми этими случаями + try/except/finally.
# ==============================================================================
def process_argument(data):
    """
    Универсальная функция обработки аргумента в зависимости от его типа.
    Включает обработку исключений и блок finally.
    """

    try:
        # Случай 1: Передан кортеж (tuple)
        if isinstance(data, tuple):
            count = len(data)
            result_info = {
                "тип": "tuple (кортеж)",
                "количество_элементов": count
            }
            print(f"[Обработка кортежа]: Количество элементов = {count}")

        # Случай 2: Передан список (list)
        elif isinstance(data, list):
            # Находим сумму элементов до первого нуля
            sum_before_zero = 0
            found_zero = False
            for elem in data:
                if not isinstance(elem, (int, float)):
                    raise TypeError(
                        f"Элемент списка '{elem}' не является числом!")
                if elem == 0:
                    found_zero = True
                    break
                sum_before_zero += elem

            status = "до первого 0" if found_zero else "всего списка (нулей нет)"
            result_info = {
                "тип": "list (список)",
                "сумма": sum_before_zero,
                "статус": status
            }
            print(f"[Обработка списка]: Сумма ({status}) = {sum_before_zero}")

        # Случай 3: Передано число (целое int или вещественное float)
        elif isinstance(data, (int, float)) and not isinstance(data, bool):
            # Реверс числа
            is_negative = data < 0
            s = str(abs(data))
            reversed_str = s[::-1]
            if isinstance(data, int):
                reversed_num = int(reversed_str)
                if is_negative:
                    reversed_num = -reversed_num
            else:
                reversed_num = float(reversed_str)
                if is_negative:
                    reversed_num = -reversed_num

            result_info = {
                "тип": "number (число)",
                "исходное": data,
                "реверс": reversed_num
            }
            print(
                f"[Обработка числа]: Исходное: {data} -> В обратном порядке: {reversed_num}")

        # Случай 4: Передана строка (str)
        elif isinstance(data, str):
            words = data.split()
            word_count = len(words)

            if not data:
                most_frequent_char = None
                max_char_count = 0
            else:
                # Подсчет самого частого символа
                char_counts = Counter(data)
                most_frequent_char, max_char_count = char_counts.most_common(1)[
                    0]

            char_display = "' '" if most_frequent_char == " " else f"'{most_frequent_char}'"
            result_info = {
                "тип": "str (строка)",
                "количество_слов": word_count,
                "самый_частый_символ": most_frequent_char,
                "частота_символа": max_char_count
            }
            print(
                f"[Обработка строки]: Слов = {word_count}, "
                f"самый частый символ = {char_display} (встретился {max_char_count} раз)"
            )

        else:
            raise TypeError(
                f"Неподдерживаемый тип данных: {
                    type(data).__name__}")

        return result_info

    except TypeError as te:
        print(f"[Исключение TypeError]: {te}")
        return {"ошибка": str(te)}
    except Exception as e:
        print(f"[Исключение общего типа]: {e}")
        return {"ошибка": str(e)}
    finally:
        print("  -> Завершение вызова функции process_argument (блок finally выполнен)")


def run_task2():
    print("=" * 60)
    print("ЗАДАНИЕ 2: Универсальная функция обработки различных типов")
    print("=" * 60)

    test_cases = [
        ("Кортеж", (10, 20, "яблоко", 45.6, True)),
        ("Список с нулем", [5, 12, 3, 0, 99, 100]),
        ("Список без нуля", [10, 20, 30]),
        ("Целое положительное число", 987654),
        ("Отрицательное число", -12345),
        ("Строка с текстом", "Machine learning and Python programming are awesome!"),
        ("Неподдерживаемый тип (множество set)", {1, 2, 3}),
        ("Список с некорректным элементом", [1, 2, "error", 0])
    ]

    for label, val in test_cases:
        print(f"\n--- Тест: {label} (Значение: {val}) ---")
        try:
            res = process_argument(val)
            print(f"Результат: {res}")
        except Exception as e:
            print(f"Внешний перехват: {e}")
    print()


# ==============================================================================
# Задание 3 (2 балла):
# Найдите индексы первого вхождения максимального элемента в двумерном массиве.
# Выведите два числа: номер строки и номер столбца, в которых стоит наибольший элемент.
# Требование: использовать функции и блок try/except/finally.
# ==============================================================================
def find_first_max_index(matrix: list):
    """
    Находит индексы (строка, столбец) первого вхождения максимального элемента в матрице.
    Возвращает кортеж (row_index, col_index, max_value).
    """
    try:
        if not isinstance(matrix, list) or len(matrix) == 0:
            raise ValueError("Матрица должна быть непустым списком списков!")

        if not all(isinstance(row, list) and len(row) > 0 for row in matrix):
            raise ValueError(
                "Все строки матрицы должны быть непустыми списками!")

        # Инициализируем максимальный элемент первым элементом матрицы
        max_val = matrix[0][0]
        max_row = 0
        max_col = 0

        # Поиск максимального элемента (первого вхождения)
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise TypeError(
                        f"Элемент [{i}][{j}] = '{val}' не является числом!")
                if val > max_val:
                    max_val = val
                    max_row = i
                    max_col = j

        return max_row, max_col, max_val

    except (ValueError, TypeError) as err:
        print(f"[Ошибка в find_first_max_index]: {err}")
        return None, None, None
    finally:
        print("  -> Завершение вызова функции find_first_max_index (блок finally выполнен)")


def run_task3():
    print("=" * 60)
    print("ЗАДАНИЕ 3: Поиск индексов первого вхождения максимума в матрице")
    print("=" * 60)

    sample_matrix = [
        [12, 45, 7, 89, 23],
        # 94 встречается дважды - первый раз в строке 1, столбце 0
        [94, 3, 94, 18, 55],
        [6, 72, 88, 11, 40]
    ]

    print("Тестовая двумерная матрица:")
    for row in sample_matrix:
        print("  ", row)

    row_idx, col_idx, max_val = find_first_max_index(sample_matrix)
    if row_idx is not None:
        print(f"\nМаксимальный элемент: {max_val}")
        print(f"Индексы (0-based): строка = {row_idx}, столбец = {col_idx}")
        print(
            f"Порядковые номера (1-based): строка №{row_idx + 1}, столбец №{col_idx + 1}")

    # Тест на обработку ошибки (пустая матрица)
    print("\nТест с некорректными данными (пустая матрица):")
    find_first_max_index([])
    print()


# ==============================================================================
# Задание 4 (2 балла):
# Напишите программу, демонстрирующую работу try\except\finally.
# ==============================================================================
def safe_calculator_demo(a, b, operation: str):
    """
    Демонстрационная функция, реализующая полное ветвление try - except - else - finally.
    Показывает:
    - перехват деления на ноль (ZeroDivisionError)
    - перехват несовместимых типов (TypeError)
    - ветку else (выполняется только при успехе)
    - ветку finally (выполняется абсолютно всегда)
    """
    print(f"\nВыполнение операции: {a} {operation} {b}")
    result = None
    try:
        print("  [1] Вход в блок try")
        if operation == "+":
            result = a + b
        elif operation == "-":
            result = a - b
        elif operation == "*":
            result = a * b
        elif operation == "/":
            result = a / b
        elif operation == "**":
            result = a ** b
        else:
            raise ValueError(f"Неизвестная операция '{operation}'!")

    except ZeroDivisionError as zde:
        print(
            f"  [2] ПЕРЕХВАЧЕНА ОШИБКА ZeroDivisionError: {zde} (деление на ноль невозможно!)")
    except TypeError as te:
        print(
            f"  [2] ПЕРЕХВАЧЕНА ОШИБКА TypeError: {te} (несовместимые типы данных!)")
    except ValueError as ve:
        print(f"  [2] ПЕРЕХВАЧЕНА ОШИБКА ValueError: {ve}")
    except Exception as e:
        print(f"  [2] ПЕРЕХВАЧЕНА ОБЩАЯ ОШИБКА: {e}")
    else:
        # Блок else выполняется только если в try не возникло ни одной ошибки
        print(
            f"  [3] Блок else: Операция выполнена успешно! Результат = {result}")
    finally:
        # Блок finally выполняется гарантированно в любом случае (даже при
        # аварии)
        print(
            "  [4] -> Завершение вызова функции safe_calculator_demo (блок finally выполнен)")

    return result


def run_task4():
    print("=" * 60)
    print("ЗАДАНИЕ 4: Полная демонстрация механизма try / except / else / finally")
    print("=" * 60)

    # 1. Успешное выполнение
    safe_calculator_demo(10, 2, "/")

    # 2. Деление на ноль
    safe_calculator_demo(25, 0, "/")

    # 3. Ошибка типа данных
    safe_calculator_demo("100", 5, "/")

    # 4. Неизвестная операция
    safe_calculator_demo(10, 5, "%&^")
    print()


def main():
    print("=" * 70)
    print("  ЛАБОРАТОРНАЯ РАБОТА №1. ЧАСТЬ 2 (ВАРИАНТ 2) - ЗАПУСК ВСЕХ ЗАДАНИЙ")
    print("=" * 70 + "\n")

    run_task1()
    run_task2()
    run_task3()
    run_task4()

    print("=" * 70)
    print("  ВЫПОЛНЕНИЕ ЧАСТИ 2 УСПЕШНО ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()
