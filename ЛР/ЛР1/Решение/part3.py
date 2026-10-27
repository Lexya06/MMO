"""
Лабораторная работа №1 по курсу "Методы машинного обучения"
Часть 3: Объектно-ориентированное программирование (ООП) в Python:
         классы, объекты, инкапсуляция, наследование, полиморфизм,
         статические переменные, методы экземпляра, @classmethod, @staticmethod
Вариант: 2
"""

import math


# ==============================================================================
# Задание 1 (1 балл):
# Класс Example. В нём пропишите 3 (метода) функции.
# Две переменные задайте статически (атрибуты класса), две динамически (в __init__).
# Первый метод: создайте локальную переменную и выведите её.
# Второй метод: верните сумму двух глобальных переменных.
# Третий метод: верните результат возведения первой динамической переменной во вторую.
# Создайте объект класса. Напечатайте работу методов. Напечатайте переменную a.
# ==============================================================================

# Глобальные переменные для Задания 1
GLOBAL_VAR_X = 15
GLOBAL_VAR_Y = 27


class Example:
    """
    Класс Example демонстрирует:
    - статические переменные (атрибуты самого класса, общие для всех объектов)
    - динамические переменные (атрибуты экземпляра, уникальные для каждого объекта)
    - локальные переменные внутри метода
    - доступ к глобальным переменным модуля
    """
    # Статические атрибуты класса (static variables)
    a = 100
    static_b = 200

    def __init__(self, dyn1: float, dyn2: float):
        # Динамические атрибуты экземпляра (динамически создаются при
        # инициализации объекта)
        self.dyn_var1 = dyn1
        self.dyn_var2 = dyn2

    def method_one(self):
        """
        Первый метод: создаёт локальную переменную внутри метода и выводит её на экран.
        """
        local_message = (
            f"Локальная переменная метода 1 (время вызова): значение = {
                self.dyn_var1 * 10}")
        print(f"  [method_one]: {local_message}")
        return local_message

    @staticmethod
    def method_two():
        """
        Второй метод: возвращает сумму 2-х глобальных переменных (GLOBAL_VAR_X + GLOBAL_VAR_Y).
        """
        result = GLOBAL_VAR_X + GLOBAL_VAR_Y
        return result

    def method_three(self):
        """
        Третий метод: возвращает результат возведения 1-й динамической переменной во 2-ю.
        """
        result = self.dyn_var1 ** self.dyn_var2
        return result


def run_task1():
    print("=" * 60)
    print("ЗАДАНИЕ 1: Работа с классом Example (статические/динамические поля и методы)")
    print("=" * 60)

    # Создаем экземпляр (объект) класса Example с dyn1=3 и dyn2=4
    example_obj = Example(dyn1=3, dyn2=4)

    print("1. Вызов первого метода:")
    example_obj.method_one()

    print("\n2. Вызов второго метода (сумма глобальных переменных):")
    sum_globals = example_obj.method_two()
    print(
        f"  GLOBAL_VAR_X ({GLOBAL_VAR_X}) + GLOBAL_VAR_Y ({GLOBAL_VAR_Y}) = {sum_globals}")

    print("\n3. Вызов третьего метода (dyn_var1 ** dyn_var2):")
    power_res = example_obj.method_three()
    print(
        f"  dyn_var1 ({example_obj.dyn_var1}) ** dyn_var2 ({example_obj.dyn_var2}) = {power_res}")

    print("\n4. Печать статической переменной 'a':")
    print(f"  Обращение через класс: Example.a = {Example.a}")
    print(f"  Обращение через объект: example_obj.a = {example_obj.a}\n")


# ==============================================================================
# Задание 2 (3 балла):
# Требуется написать программу, которая вычисляет общую площадь стены комнаты,
# которую необходимо оклеить обоями. При этом окна, двери, пол и потолок
# оклеивать не нужно.
# Реализуем задачу в объектно-ориентированном стиле.
# ==============================================================================
class Window:
    """Класс для представления оконного проема"""

    def __init__(self, width: float, height: float, count: int = 1):
        self.width = width
        self.height = height
        self.count = count

    def get_area(self) -> float:
        """Возвращает суммарную площадь окон данного типа"""
        return self.width * self.height * self.count


class Door:
    """Класс для представления дверного проема"""

    def __init__(self, width: float, height: float, count: int = 1):
        self.width = width
        self.height = height
        self.count = count

    def get_area(self) -> float:
        """Возвращает суммарную площадь дверей данного типа"""
        return self.width * self.height * self.count


class Room:
    """
    Класс Room моделирует комнату с заданными габаритами и списком проемов
    (окон и дверей), позволяя рассчитать чистую площадь стен для оклейки обоями.
    """

    def __init__(self, length: float, width: float, height: float):
        if length <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                "Габариты комнаты должны быть положительными числами!")
        self.length = length
        self.width = width
        self.height = height
        self.windows: list[Window] = []  # Список окон
        self.doors: list[Door] = []      # Список дверей

    def add_window(self, width: float, height: float, count: int = 1):
        """Добавить окно (или несколько одинаковых окон) в комнату"""
        self.windows.append(Window(width, height, count))

    def add_door(self, width: float, height: float, count: int = 1):
        """Добавить дверь (или несколько одинаковых дверей) в комнату"""
        self.doors.append(Door(width, height, count))

    def get_total_wall_area(self) -> float:
        """
        Вычисляет полную площадь 4 стен без вычета проемов:
        S = 2 * (длина + ширина) * высота
        """
        perimeter = 2 * (self.length + self.width)
        return perimeter * self.height

    def get_unpasted_area(self) -> float:
        """Вычисляет суммарную площадь окон и дверей (которые не оклеиваются)"""
        windows_area = sum(w.get_area() for w in self.windows)
        doors_area = sum(d.get_area() for d in self.doors)
        return windows_area + doors_area

    def get_wallpaper_area(self) -> float:
        """
        Вычисляет чистую рабочую площадь стен под поклейку обоев:
        S_чистая = S_стен - S_окон - S_дверей
        """
        net_area = self.get_total_wall_area() - self.get_unpasted_area()
        return max(0.0, net_area)

    def calculate_wallpaper_rolls(
        self,
        roll_width: float = 0.53,
        roll_length: float = 10.05,
        reserve_percent: float = 10.0
    ) -> int:
        """
        Расчет количества стандартных рулонов обоев с учетом запаса на подгонку.
        """
        roll_area = roll_width * roll_length
        needed_area = self.get_wallpaper_area() * (1 + reserve_percent / 100)
        return math.ceil(needed_area / roll_area)


def run_task2():
    print("=" * 60)
    print("ЗАДАНИЕ 2: Расчет площади оклейки стен обоями (Класс Room)")
    print("=" * 60)

    # Задаем размеры комнаты: длина = 6.0 м, ширина = 4.0 м, высота = 2.75 м
    my_room = Room(length=6.0, width=4.0, height=2.75)

    # Добавляем 2 окна (ширина 1.5 м, высота 1.4 м)
    my_room.add_window(width=1.5, height=1.4, count=2)

    # Добавляем 1 дверь (ширина 0.9 м, высота 2.0 м)
    my_room.add_door(width=0.9, height=2.0, count=1)

    # Расчет площадей
    total_walls = my_room.get_total_wall_area()
    openings_area = my_room.get_unpasted_area()
    wallpaper_area = my_room.get_wallpaper_area()
    rolls = my_room.calculate_wallpaper_rolls(
        roll_width=0.53,
        roll_length=10.05,
        reserve_percent=10.0
    )

    print("Характеристики комнаты:")
    print(
        f"  • Габариты: Длина={
            my_room.length} м, Ширина={
            my_room.width} м, Высота={
                my_room.height} м")
    print(
        f"  • Окна: 2 шт. (1.5 x 1.4 м, площадь = {
            1.5 *
            1.4 *
            2:.2f} кв. м)")
    print(
        f"  • Двери: 1 шт. (0.9 x 2.0 м, площадь = {
            0.9 *
            2.0 *
            1:.2f} кв. м)")
    print("-" * 50)
    print(f"  -> Полная площадь 4-х стен: {total_walls:.2f} кв. м")
    print(
        f"  -> Площадь не оклеиваемых проемов (окна + двери): {openings_area:.2f} кв. м")
    print(
        f"  -> ЧИСТАЯ ПЛОЩАДЬ СТЕН ДЛЯ ОКЛЕЙКИ ОБОЯМИ: {wallpaper_area:.2f} кв. м")
    print(
        f"  -> Необходимое количество рулонов обоев (0.53x10.05м, запас 10%): {rolls} шт.\n")


# ==============================================================================
# Задание 3 (3 балла):
# 1. Реализуйте базовый класс Car.
# 2. Атрибуты: speed, color, name, is_police (булево).
#    Методы: go, stop, turn(direction), которые сообщают, что машина поехала,
#    остановилась, повернула (куда).
# 3. Опишите несколько дочерних классов: TownCar, SportCar, WorkCar, PoliceCar.
# 4. Добавьте в базовый класс метод show_speed, показывающий текущую скорость.
# 5. Для классов TownCar и WorkCar переопределите метод show_speed: при значении
#    скорости свыше 60 (TownCar) и 40 (WorkCar) должно выводиться сообщение
#    о превышении скорости.
# ==============================================================================
class Car:
    """
    Базовый класс Car (Автомобиль).
    Демонстрирует базовые методы управления и отображения скорости.
    """

    def __init__(
            self,
            speed: float,
            color: str,
            name: str,
            is_police: bool = False):
        self.speed = speed
        self.color = color
        self.name = name
        self.is_police = is_police

    def go(self):
        """Метод начала движения"""
        print(f"Автомобиль '{self.name}' ({self.color}) начал движение.")

    def stop(self):
        """Метод остановки"""
        self.speed = 0
        print(f"Автомобиль '{self.name}' полностью остановился.")

    def turn(self, direction: str):
        """Метод поворота"""
        print(f"Автомобиль '{self.name}' совершил поворот {direction}.")

    def show_speed(self):
        """Базовый метод отображения скорости"""
        print(f"Текущая скорость автомобиля '{self.name}': {self.speed} км/ч.")


class TownCar(Car):
    """
    Городской автомобиль (дочерний класс).
    Ограничение скорости: 60 км/ч.
    """

    def __init__(self, speed: float, color: str, name: str):
        super().__init__(speed, color, name, is_police=False)

    def show_speed(self):
        """Переопределенный метод контроля скорости для TownCar"""
        print(
            f"Городской авто '{
                self.name}': текущая скорость = {
                self.speed} км/ч.")
        if self.speed > 60:
            print(
                f"  ВНИМАНИЕ: Превышение допустимой скорости в городе (60 км/ч)! "
                f"Скорость: {self.speed} км/ч!"
            )


class SportCar(Car):
    """Спортивный автомобиль (дочерний класс)"""

    def __init__(self, speed: float, color: str, name: str):
        super().__init__(speed, color, name, is_police=False)


class WorkCar(Car):
    """
    Рабочий / грузовой автомобиль (дочерний класс).
    Ограничение скорости: 40 км/ч.
    """

    def __init__(self, speed: float, color: str, name: str):
        super().__init__(speed, color, name, is_police=False)

    def show_speed(self):
        """Переопределенный метод контроля скорости для WorkCar"""
        print(
            f"Рабочий авто '{
                self.name}': текущая скорость = {
                self.speed} км/ч.")
        if self.speed > 40:
            print(
                f"  ВНИМАНИЕ: Превышение допустимой скорости для рабочего транспорта (40 км/ч)! "
                f"Скорость: {self.speed} км/ч!"
            )


class PoliceCar(Car):
    """Полицейский автомобиль (дочерний класс)"""

    def __init__(self, speed: float, color: str, name: str):
        super().__init__(speed, color, name, is_police=True)


def run_task3():
    print("=" * 60)
    print("ЗАДАНИЕ 3: Иерархия классов автомобилей (Наследование и полиморфизм)")
    print("=" * 60)

    # Создаем объекты каждого типа автомобиля
    town_car = TownCar(speed=65, color="синий", name="Skoda Octavia")
    sport_car = SportCar(speed=180, color="красный", name="Ferrari F8")
    work_car = WorkCar(speed=48, color="желтый", name="Трактор JCB")
    police_car = PoliceCar(
        speed=90,
        color="бело-синий",
        name="Ford Police Interceptor")

    fleet = [town_car, sport_car, work_car, police_car]

    for car in fleet:
        print(
            f"\n--- Тестирование автомобиля: {car.name} ({car.__class__.__name__}) ---")
        print(
            f"Цвет: {
                car.color} | Полицейский: {
                'Да' if car.is_police else 'Нет'}")
        car.go()
        car.turn("налево")
        car.show_speed()
        car.stop()
    print()


# ==============================================================================
# Задание 4 (1-3 балла):
# Придумать класс самостоятельно, реализовать в нем:
# 1) методы экземпляра класса (instance methods)
# 2) статические методы (@staticmethod)
# 3) методы класса (@classmethod)
# ==============================================================================
class BankAccount:
    """
    Пользовательский класс 'Банковский счет' (BankAccount), демонстрирующий:
    1. Методы экземпляра (работают с конкретным счетом через self)
    2. Методы класса @classmethod (работают с состоянием класса BankAccount через cls)
    3. Статические методы @staticmethod (утилитарные функции, не привязанные к состоянию)
    """

    # Атрибуты уровня класса (общие для всех счетов банка)
    bank_name = "Белорусский Инновационный Банк"
    interest_rate_annual = 4.5  # Годовая процентная ставка по депозитам (%)
    total_accounts_opened = 0   # Общее количество открытых счетов

    def __init__(
            self,
            account_holder: str,
            initial_balance: float = 0.0,
            currency: str = "BYN"):
        # Валидация входных данных через статический метод
        if not self.is_valid_currency(currency):
            raise ValueError(f"Валюта '{currency}' не поддерживается банком!")
        if initial_balance < 0:
            raise ValueError("Начальный баланс не может быть отрицательным!")

        # Атрибуты конкретного экземпляра
        self.account_holder = account_holder
        self.balance = float(initial_balance)
        self.currency = currency.upper()

        # Увеличиваем счетчик открытых счетов в банке
        BankAccount.total_accounts_opened += 1
        self.account_id = f"ACC-{BankAccount.total_accounts_opened:04d}"

    # ----------------- 1. Методы экземпляра (Instance Methods) --------------
    def deposit(self, amount: float):
        """Пополнение счета (метод экземпляра)"""
        if amount <= 0:
            raise ValueError("Сумма пополнения должна быть больше 0!")
        self.balance += amount
        print(
            f"[{self.account_id}] Счёт пополнен на {amount:.2f} {self.currency}. "
            f"Текущий баланс: {self.balance:.2f} {self.currency}"
        )
        return self.balance

    def withdraw(self, amount: float):
        """Снятие средств со счета (метод экземпляра)"""
        if amount <= 0:
            raise ValueError("Сумма снятия должна быть положительной!")
        if amount > self.balance:
            print(
                f"[{self.account_id}] Ошибка: недостаточно средств! "
                f"Баланс: {self.balance:.2f} {self.currency}, запрошено: {amount:.2f} {self.currency}"
            )
            return False
        self.balance -= amount
        print(
            f"[{self.account_id}] Снято {amount:.2f} {self.currency}. "
            f"Остаток на счете: {self.balance:.2f} {self.currency}"
        )
        return True

    def calculate_annual_interest(self) -> float:
        """Расчет годового дохода по процентам для текущего счета (метод экземпляра)"""
        earned = self.balance * (BankAccount.interest_rate_annual / 100.0)
        return earned

    def get_info(self) -> str:
        """Формирование строки с информацией о счете"""
        return (
            f"Счет {self.account_id} | Владелец: {self.account_holder:<15} | "
            f"Баланс: {self.balance:>10.2f} {self.currency}"
        )

    # ------------------- 2. Методы класса (@classmethod) -------------------
    @classmethod
    def set_interest_rate(cls, new_rate: float):
        """
        Метод класса: изменяет базовую процентную ставку для всех счетов банка.
        """
        if new_rate < 0:
            raise ValueError("Процентная ставка не может быть отрицательной!")
        old_rate = cls.interest_rate_annual
        cls.interest_rate_annual = new_rate
        print(
            f"[{cls.bank_name}] Годовая ставка изменена: {old_rate}% -> {cls.interest_rate_annual}%")

    @classmethod
    def create_from_string(cls, data_str: str):
        """
        Альтернативный конструктор (фабричный метод класса):
        Создает объект BankAccount из строки формата 'ФИО, Баланс, Валюта'
        """
        parts = [p.strip() for p in data_str.split(",")]
        if len(parts) != 3:
            raise ValueError(
                "Неверный формат строки! Ожидается 'ФИО, баланс, валюта'")
        holder = parts[0]
        balance = float(parts[1])
        curr = parts[2]
        return cls(
            account_holder=holder,
            initial_balance=balance,
            currency=curr)

    # ----------------- 3. Статические методы (@staticmethod) -----------------
    @staticmethod
    def is_valid_currency(currency_code: str) -> bool:
        """
        Статический метод: проверяет, является ли валюта допустимой.
        """
        allowed_currencies = {"BYN", "RUB", "USD", "EUR", "CNY"}
        return currency_code.upper() in allowed_currencies



def run_task4():
    print("=" * 60)
    print("ЗАДАНИЕ 4: Самостоятельный класс BankAccount (@classmethod, @staticmethod, instance methods)")
    print("=" * 60)

    print(f"Банк: {BankAccount.bank_name}")
    print(
        f"Текущая процентная ставка: {
            BankAccount.interest_rate_annual}% годовых\n")

    # 1. Демонстрация статического метода (@staticmethod)
    print("1. Проверка допустимости валют через @staticmethod:")
    print(
        f"  Валюта 'BYN' поддерживается? {
            BankAccount.is_valid_currency('BYN')}")
    print(
        f"  Валюта 'BTC' поддерживается? {
            BankAccount.is_valid_currency('BTC')}")

    # 2. Демонстрация метода класса (альтернативный конструктор @classmethod)
    print("\n2. Создание счета через фабричный метод класса (@classmethod create_from_string):")
    acc1 = BankAccount.create_from_string("Иванов Иван Иванович, 1500.50, BYN")
    print(f"  Создан: {acc1.get_info()}")

    # Создание второго счета стандартным конструктором
    acc2 = BankAccount("Петрова Анна Сергеевна", 3200.0, "USD")
    print(f"  Создан: {acc2.get_info()}")

    # 3. Демонстрация методов экземпляра (instance methods)
    print("\n3. Выполнение операций по счетам (методы экземпляра):")
    acc1.deposit(500.0)
    acc1.withdraw(200.0)
    acc1.withdraw(5000.0)  # Попытка снять больше баланса

    interest_1 = acc1.calculate_annual_interest()
    print(
        f"  Ожидаемый годовой доход по процентам ({
            BankAccount.interest_rate_annual}%): " f"{
            interest_1:.2f} {
                acc1.currency}")

    # 4. Демонстрация изменения параметра класса через @classmethod
    print("\n4. Изменение ставки банка через @classmethod:")
    BankAccount.set_interest_rate(6.0)
    new_interest_1 = acc1.calculate_annual_interest()
    print(
        f"  Новый годовой доход по процентам ({
            BankAccount.interest_rate_annual}%): " f"{
            new_interest_1:.2f} {
                acc1.currency}")

    print(
        f"\nВсего открыто счетов в банке: {
            BankAccount.total_accounts_opened}\n")


def main():
    print("=" * 70)
    print("  ЛАБОРАТОРНАЯ РАБОТА №1. ЧАСТЬ 3 (ВАРИАНТ 2) - ЗАПУСК ВСЕХ ЗАДАНИЙ")
    print("=" * 70 + "\n")

    run_task1()
    run_task2()
    run_task3()
    run_task4()

    print("=" * 70)
    print("  ВЫПОЛНЕНИЕ ЧАСТИ 3 УСПЕШНО ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()
