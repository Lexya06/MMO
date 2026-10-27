# -*- coding: utf-8 -*-
"""
Скрипт для сборки академического отчета по лабораторной работе № 2 в формате LaTeX (report.tex).
Формирует документ в строгом соответствии с СТП 01-2017 и шаблоном LR1:
- Точно такие же пакеты и геометрия страницы (Times New Roman / tempora, 14pt, поля 30/10/20/20)
- Титульный лист БГУИР без слова "вариант"
- Полная математическая теория с формулами
- Включение всех модулей исходного кода в окружениях CodeListing
- Вставка всех 10 аналитических графиков и таблиц
- Развернутые выводы
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOL_DIR = BASE_DIR
REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Отчет'))
OVERLEAF_DIR = os.path.join(REPORT_DIR, 'Отчет_Overleaf')
OUT_TEX_PATH = os.path.join(OVERLEAF_DIR, 'report.tex')

os.makedirs(OVERLEAF_DIR, exist_ok=True)


def read_file(name: str) -> str:
    with open(os.path.join(SOL_DIR, name), 'r', encoding='utf-8') as f:
        code = f.read().strip()

    # Точно так же, как в ЛР1 (update_latex.py):
    for ch in ['•', '—', '–', '«', '»']:
        code = code.replace(ch, '*' if ch == '•' else ('"' if ch in ['«', '»'] else '-'))

    code = code.replace('№', 'No. ')
    code = code.replace('✓', '[OK]')
    code = code.replace('≥', '>=')
    code = code.replace('≤', '<=')
    code = code.replace('≈', '~')
    code = code.replace('μ', 'mean')
    code = code.replace('σ', 'sigma')

    # Заменяем обратные слэши на прямые (как в update_latex.py ЛР1), чтобы исключить любые конфликты fvextra/fancyvrb
    code = code.replace('\\', '/')
    return code


config_code = read_file('config.py')
data_loader_code = read_file('data_loader.py')
preprocessor_code = read_file('preprocessor.py')
fe_code = read_file('feature_engineering.py')
scaler_code = read_file('scaler.py')
visualizer_code = read_file('visualizer.py')
main_code = read_file('main.py')
solution_code = read_file('solution.py')

header = r'''\documentclass[14pt,a4paper]{extarticle}

% --- 1. Поддержка кириллицы, кодировок и шрифтов (СТП 01-2017) ---
\usepackage{cmap}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[russian]{babel}
\usepackage{tempora} % Истинный Times New Roman для кириллицы и латиницы (с полной поддержкой Bold/Italic)

% --- 2. Геометрия страницы по СТП 01-2017 и образцу ОтчетRest.docx (Левое 30мм, Правое 15мм, Верхнее 20мм, Нижнее 20мм) ---
\usepackage[a4paper, left=30mm, right=15mm, top=20mm, bottom=20mm]{geometry}

% --- 3. Межстрочный интервал (полуторный) и абзацный отступ (1.25 см) ---
\usepackage{setspace}
\setstretch{1.25}
\usepackage{indentfirst}
\setlength{\parindent}{1.25cm}
\setlength{\parskip}{0pt}

% --- 4. Предотвращение выезда длинных слов и идентификаторов за границы листа ---
\sloppy
\emergencystretch=2.5em
\righthyphenmin=2
\lefthyphenmin=2

% --- 5. Математические пакеты и таблицы ---
\usepackage{amsmath,amssymb}
\usepackage{array}
\usepackage{tabularx}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}

% --- 6. Оформление подписей к рисункам и таблицам по СТП 01-2017 (длинное тире) ---
\DeclareCaptionLabelSeparator{emdash}{~---~}
\captionsetup[figure]{name=Рисунок, labelsep=emdash, justification=centering, font=normalfont, size=normalsize}
\captionsetup[table]{name=Таблица, labelsep=emdash, justification=raggedright, singlelinecheck=false, font=normalfont, size=normalsize}

% --- 7. Оформление заголовков разделов и подразделов по СТП 01-2017 (14pt, абзацный отступ 1.25 см) ---
\usepackage{titlesec}
\titleformat{\section}[block]{\bfseries\normalsize\raggedright}{\thesection}{1em}{\MakeUppercase}
\titleformat{\subsection}[block]{\bfseries\normalsize\raggedright}{\thesubsection}{1em}{}
\titleformat{\subsubsection}[block]{\bfseries\normalsize\raggedright}{\thesubsubsection}{1em}{}

\titlespacing*{\section}{\parindent}{18pt}{18pt}
\titlespacing*{\subsection}{\parindent}{18pt}{18pt}
\titlespacing*{\subsubsection}{\parindent}{18pt}{18pt}

% --- 8. Листинги кода по СТП 01-2017 (без номеров строк слева, с рамкой и переносом) ---
\usepackage{fvextra}
\usepackage{fancyvrb}
\DefineVerbatimEnvironment{CodeListing}{Verbatim}{
    fontsize=\fontsize{9.5pt}{11.5pt}\selectfont,
    framesep=2.5mm,
    framerule=0.5pt,
    breaklines=true,
    breaksymbol={\tiny\ensuremath{\hookrightarrow}},
    xleftmargin=0pt,
    xrightmargin=0pt
}

% --- 9. Списки: отступы и стиль маркированных списков (маркер-дефис -) ---
\usepackage{enumitem}
\setlist{nosep, leftmargin=1.25cm}
\setlist[enumerate]{label=\arabic*., leftmargin=1.25cm}
\setlist[enumerate,1]{label=\arabic*., leftmargin=1.25cm}
\setlist[enumerate,2]{label=\arabic{enumi}.\arabic*., leftmargin=1.25cm}
\setlist[itemize]{label=-, leftmargin=1.25cm}
\renewcommand{\labelitemi}{-}
\renewcommand{\labelitemii}{-}

% --- 10. Нумерация страниц справа по СТП 01-2017 ---
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\fancypagestyle{plain}{
    \fancyhf{}
    \fancyfoot[R]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
}

\begin{document}

% ==============================================================================
% ТИТУЛЬНЫЙ ЛИСТ (образцу СТП 01-2017 и ЛР1, без слова вариант)
% ==============================================================================
\begin{titlepage}
    \begin{center}
        Министерство образования Республики Беларусь\\[0.15cm]
        Учреждение образования\\[0.15cm]
        «БЕЛОРУССКИЙ ГОСУДАРСТВЕННЫЙ УНИВЕРСИТЕТ\\
        ИНФОРМАТИКИ И РАДИОЭЛЕКТРОНИКИ»
    \end{center}
    
    \vspace{0.4cm}
    
    \begin{flushleft}
        Факультет компьютерных систем и сетей\\[0.1cm]
        Кафедра программного обеспечения информационных технологий\\[0.1cm]
        Дисциплина: Методы машинного обучения (ММО)
    \end{flushleft}
    
    \vspace{2.2cm}
    
    \begin{center}
        \large\textbf{ОТЧЕТ}\\[0.2cm]
        \normalsize
        {по лабораторной работе № 2}\\[0.3cm]
        Тема работы: Разведочный и предиктивный анализ данных
    \end{center}
    
    \vspace{2.5cm}
    
    \noindent
    \begin{tabular}{@{}p{0.62\textwidth}@{}p{0.38\textwidth}@{}}
        Выполнила: & \\
        студентка гр. 351002 & Алексейчик Я.\,Н. \\[0.5cm]
        Проверил: & Лебедевич А.\,В.
    \end{tabular}
    
    \vfill
    
    \begin{center}
        Минск 2026
    \end{center}
\end{titlepage}

\setcounter{page}{2}

% ==============================================================================
% 1 ЦЕЛЬ ВЫПОЛНЕНИЯ РАБОТЫ
% ==============================================================================
\section{Цель выполнения лабораторной работы}

Целью выполнения лабораторной работы является практическое освоение современных методов сбора, первичного разведочного анализа (Exploratory Data Analysis, EDA) и комплексной предобработки структурированных табличных данных для задач машинного обучения с использованием экосистемы Python (\texttt{pandas}, \texttt{numpy}, \texttt{scikit-learn}, \texttt{matplotlib}, \texttt{seaborn}):

\begin{enumerate}
    \item \textbf{Импорт и первичный структурный анализ выборки:} исследование схемы данных, определение размерности, разделение типов признаков на числовые и категориальные, вычисление фундаментальных параметров описательной статистики (математическое ожидание, дисперсия, квартили, размах).
    \item \textbf{Анализ и обработка пропущенных значений:} статистическая детекция скрытых пропусков (\texttt{NaN}), их визуализация тепловыми картами и столбчатыми диаграммами, обоснование и реализация математически корректной стратегии импутации модой для категориальных шкал с целью сохранения 100\% объема наблюдений; контрольная графическая верификация чистоты данных.
    \item \textbf{Обнаружение и устранение аномальных значений (выбросов):} расчет границ допустимого диапазона по правилу межквартильного размаха Дж. Тьюки ($1.5 \times IQR$), выявление логических противоречий в данных, визуализация аномалий на гистограммах и диаграммах размаха BoxPlot; устранение выбросов методом винсоризации (\textit{clipping}) без удаления строк; графическое подтверждение отсутствия выбросов.
    \item \textbf{Генерация новых признаков (Feature Engineering):} математический синтез 5 предметно-обоснованных характеристик (отношений, линейных взвешенных сверток, разностей прогресса и бинарных пороговых индикаторов) для обогащения признакового пространства и раскрытия нелинейных взаимосвязей.
    \item \textbf{Кодирование категориальных признаков:} приведение строковых данных к числовому матричному виду посредством трех методов: бинаризации дихотомических признаков (0 / 1), порядкового кодирования (Ordinal Encoding) для ранговых шкал (0, 1, 2) и прямого кодирования (One-Hot Encoding, OHE) для номинальных переменных с целью исключения фиктивного метрического порядка.
    \item \textbf{Масштабирование числовых признаков:} программная реализация нормализации (\texttt{MinMaxScaler}) в единичный диапазон $[0, 1]$ и стандартизации (\texttt{StandardScaler}) к нулевому среднему и единичной дисперсии; аналитическое и графическое сопоставление плотностей распределения (KDE), доказательство инвариантности топологии распределений при линейных трансформациях.
\end{enumerate}

% ==============================================================================
% 2 КРАТКИЕ ТЕОРЕТИЧЕСКИЕ СВЕДЕНИЯ
% ==============================================================================
\clearpage
\section{Краткие теоретические сведения}

\subsection{Концепция Exploratory Data Analysis (EDA) и предобработки данных}

В прикладном машинном обучении качество и надежность прогнозных моделей напрямую определяются качеством входных обучающих выборок. Принцип \textbf{«Garbage In --- Garbage Out»} (GIGO) констатирует, что даже архитектурно совершенный классификатор или регрессор при обучении на зашумленных данных с пропусками, несогласованными масштабами и выбросами выявит ложные корреляции и продемонстрирует низкую обобщающую способность.

Разведочный анализ данных (Exploratory Data Analysis, EDA) представляет собой формализованную методологию критического анализа выборок до построения предиктивных алгоритмов. EDA решает следующие ключевые задачи:
\begin{itemize}
    \item выявление внутренней геометрической структуры признакового пространства;
    \item локализация пропущенных значений и скрытых искажений;
    \item обнаружение экстремальных аномалий, деформирующих оптимизационный ландшафт функций потерь;
    \item преобразование качественных шкал в согласованные числовые тензоры;
    \item выравнивание масштабных коэффициентов признаков для обеспечения численной стабильности градиентных методов.
\end{itemize}

\subsection{Анализ и методы импутации пропущенных значений}

Пропуски в данных (\textit{Missing Values}) возникают вследствие сбоев опрашивающих систем, ошибок операторов, нежелания респондентов указывать конфиденциальные данные или потери сетевых пакетов. В математической статистике выделяют три фундаментальных механизма возникновения пропусков:
\begin{enumerate}
    \item \textbf{MCAR (Missing Completely at Random):} вероятность пропуска значения не зависит ни от наблюдаемых, ни от ненаблюдаемых признаков.
    \item \textbf{MAR (Missing at Random):} вероятность отсутствия значения признака $X$ систематически связана с другими известными признаками $Y$, но не с истинной величиной самого $X$.
    \item \textbf{MNAR (Missing Not at Random):} вероятность пропуска непосредственно детерминирована ненаблюдаемой величиной самого признака.
\end{enumerate}

\subsubsection{Стратегии обработки пропусков:}
\begin{itemize}
    \item \textbf{Полное удаление наблюдений (Listwise Deletion):} удаление любой строки, содержащей хотя бы один пропуск. Метод допустим только при доле поврежденных записей $<1\%$ и механизме MCAR. В противном случае происходит катастрофическая деградация статистической мощности выборки и смещение оценок.
    \item \textbf{Импутация средним или медианой:} эффективна для непрерывных числовых величин с нормальным или скошенным распределением соответственно.
    \item \textbf{Импутация модой (Mode Imputation):} замещение пропущенных значений наиболее частой категорией выборки. Для категориальной случайной величины $X$, принимающей значения из дискретного множества $C = \{c_1, \dots, c_k\}$, мода определяется как аргумент максимума эмпирической частоты:
    \begin{equation}
        M_0 = \arg\max_{c \in C} \sum_{i=1}^N \mathbb{I}(x_i = c),
    \end{equation}
    где $\mathbb{I}(\cdot)$ --- индикаторная функция. При работе с категориальными шкалами мода является единственной математически корректной мерой центральной тенденции, поскольку операции суммирования и деления для качественных меток не определены.
\end{itemize}

\subsection{Математический аппарат обнаружения выбросов по методу IQR}

\textbf{Выброс (Outlier)} --- это наблюдение, находящееся на аномально большом удалении от математического ожидания основной генеральной совокупности. Выбросы искажают оценки выборочной дисперсии, смещают разделяющие гиперплоскости линейных моделей и вызывают расходимость градиентного спуска.

Для устойчивого (\textit{robust}) поиска аномалий в распределениях применяется критерий Джона Тьюки, основанный на квантилях и \textbf{межквартильном размахе (Interquartile Range, IQR)}:
\begin{enumerate}
    \item Наблюдения упорядочиваются по возрастанию.
    \item Вычисляются первый квартиль $Q_1$ (25-й перцентиль) и третий квартиль $Q_3$ (75-й перцентиль):
    \begin{equation}
        P(X \le Q_1) = 0.25, \quad P(X \le Q_3) = 0.75.
    \end{equation}
    \item Межквартильный размах задает ширину диапазона, охватывающего 50\% центральной массы выборки:
    \begin{equation}
        IQR = Q_3 - Q_1.
    \end{equation}
    \item Границы статистически допустимых значений («усы» диаграммы размаха BoxPlot) определяются как:
    \begin{equation}
        X_{\text{нижн}} = Q_1 - 1.5 \times IQR, \quad X_{\text{верх}} = Q_3 + 1.5 \times IQR.
    \end{equation}
\end{enumerate}

Коэффициент 1.5 выбран исходя из свойств стандартного нормального распределения $\mathcal{N}(0, \sigma^2)$, где интервал $[Q_1 - 1.5 \times IQR, \; Q_3 + 1.5 \times IQR]$ соответствует диапазону $[\mu - 2.698\sigma, \; \mu + 2.698\sigma]$ и охватывает 99.3\% выборки, отсекая лишь крайние 0.7\% истинных хвостов.

\subsection{Методология устранения выбросов: винсоризация против удаления}

Выделяют два фундаментальных класса аномалий:
\begin{enumerate}
    \item \textbf{Логические ошибки ввода:} значения, нарушающие аксиоматику предметной области (например, балл за экзамен $101$ при допустимой шкале $[0, 100]$). Подлежат жесткой коррекции к предельно допустимому значению шкалы.
    \item \textbf{Экстремальные статистические выбросы:} реальные, но редкие наблюдения (например, 44 часа самостоятельной учебы в неделю при норме 20).
\end{enumerate}

Физическое удаление строк с выбросами (\textit{Trimming}) уменьшает размер выборки и удаляет ценную информацию по другим признакам данных студентов. В качестве математически обоснованной альтернативы применяется \textbf{винсоризация (Winsorization / Clipping)}: экстремальные значения за пределами интервала $[X_{\text{нижн}}, X_{\text{верх}}]$ принудительно ограничиваются граничными порогами:
\begin{equation}
    x_{\text{new}} = 
    \begin{cases}
        X_{\text{нижн}}, & \text{если } x < X_{\text{нижн}}, \\
        X_{\text{верх}}, & \text{если } x > X_{\text{верх}}, \\
        x, & \text{если } X_{\text{нижн}} \le x \le X_{\text{верх}}.
    \end{cases}
\end{equation}
Винсоризация гарантирует 100\% сохранность объема обучающей выборки, ликвидирует длинные тяжелые хвосты и предотвращает взрыв градиентов при обучении моделей.

\subsection{Инженерия признаков (Feature Engineering)}

\textbf{Feature Engineering} --- это целенаправленное конструирование новых математических атрибутов на основе исходных переменных выборки с использованием предметных закономерностей. Введение явных функциональных преобразований позволяет:
\begin{itemize}
    \item выразить относительные пропорции и баланс распределения ресурсов (отношение учебы ко сну);
    \item объединить гетерогенные характеристики в единый комплексный взвешенный индекс усилий;
    \item зафиксировать динамику изменения процесса (прирост экзаменационного балла относительно прошлых тестов);
    \item дискретизировать ключевые пороговые эффекты (индикатор высокой посещаемости).
\end{itemize}

\subsection{Математические методы кодирования категориальных данных}

Компьютерные алгоритмы машинного обучения оперируют матрицами вещественных чисел. Преобразование текстовых категорий требует строгого учета характера статистической шкалы:

\begin{enumerate}
    \item \textbf{Бинаризация (Binary Encoding):} применяется для дихотомических признаков, принимающих ровно два взаимоисключающих значения ($C = \{c_0, c_1\}$):
    \begin{equation}
        f(x) = \begin{cases} 0, & \text{если } x = c_0, \\ 1, & \text{если } x = c_1. \end{cases}
    \end{equation}
    \item \textbf{Порядковое кодирование (Ordinal Encoding):} применяется для ранговых шкал, обладающих естественным монотонным отношением порядка ($c_0 < c_1 < \dots < c_{k-1}$):
    \begin{equation}
        f(c_j) = j, \quad j \in \{0, 1, \dots, k-1\}.
    \end{equation}
    Метод сохраняет транзитивность отношений и порядок интенсивности признака.
    \item \textbf{Прямое унитарное кодирование (One-Hot Encoding, OHE):} применяется для номинальных признаков, категории которых равноправны и не имеют физического порядка (например, тип влияния сверстников). Номинальный признак с $k$ уникальными градациями раскладывается на $k$ бинарных векторных индикаторов:
    \begin{equation}
        \mathbf{e}_j = (0, \dots, 0, \underbrace{1}_{j\text{-я позиция}}, 0, \dots, 0)^T.
    \end{equation}
    OHE исключает ложное предположение линейных моделей о том, что категория с кодом 2 «больше» категории с кодом 1.
\end{enumerate}

\subsection{Масштабирование числовых признаков: нормализация и стандартизация}

В работе исследованы и применены два метода масштабирования:
\begin{enumerate}
    \item \textbf{Нормализация (Min-Max Scaling):} линейное сжатие непрерывной шкалы строго в отрезок $[0, 1]$:
    \begin{equation}
        X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}.
    \end{equation}
    \item \textbf{Стандартизация (Z-Score Standardization):} аффинное центрирование признака к нулевому выборочному среднему ($\mu = 0$) и масштабирование к единичному стандартному отклонению ($\sigma = 1$):
    \begin{equation}
        X_{\text{std}} = \frac{X - \mu}{\sigma}, \quad \mu = \frac{1}{N}\sum_{i=1}^N x_i, \quad \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}.
    \end{equation}
\end{enumerate}

\textbf{Теорема об инвариантности плотности распределения:} Пусть случайная величина $X$ имеет непрерывную функцию плотности вероятностей $p_X(x)$. Линейное преобразование $Y = aX + b$ (где $a > 0$) формирует новую случайную величину $Y$ с плотностью:
\begin{equation}
    p_Y(y) = \frac{1}{a} p_X\left(\frac{y - b}{a}\right).
\end{equation}
Поскольку формулы \texttt{MinMaxScaler} ($a = \frac{1}{X_{\max}-X_{\min}}$, $b = -\frac{X_{\min}}{X_{\max}-X_{\min}}$) и \texttt{StandardScaler} ($a = \frac{1}{\sigma}$, $b = -\frac{\mu}{\sigma}$) представляют собой строго линейные преобразования с положительным множителем $a > 0$, они выполняют лишь масштабное сжатие и параллельный перенос вдоль оси абсцисс. Коэффициент асимметрии (\textit{Skewness}), эксцесс (\textit{Kurtosis}) и геометрическая форма ядерной оценки плотности (KDE) остаются абсолютно инвариантными.

% ==============================================================================
% 3 ФОРМУЛИРОВКА ЗАДАНИЯ
% ==============================================================================
\clearpage
\section{Формулировка задания}

В соответствии с официальной рабочей программой дисциплины «Методы машинного обучения» кафедры ПОИТ сформулировано следующее комплексное задание на лабораторную работу:

\begin{enumerate}
    \item \textbf{Импортировать датасет.} Выбрать исходный структурированный массив данных (самостоятельно с портала \texttt{kaggle.com} либо из прикрепленных файлов учебно-методического комплекса).
    \item \textbf{Провести очистку и разведочный анализ:}
    \begin{enumerate}
        \item Проверить данные на пропуски и наличие аномальных значений с использованием статистических критериев и графических средств визуализации (графики распределений, столбчатые диаграммы, графики типа BoxPlot, тепловые карты).
        \item Заполнить пропуски научно обоснованными методами и устранить выявленные выбросы.
        \item Графически подтвердить полное отсутствие пропусков и аномальных значений после выполнения процедур очистки.
    \end{enumerate}
    \item \textbf{Инженерия признаков и кодирование:}
    \begin{enumerate}
        \item Сгенерировать новые признаки (не менее двух предметно-обоснованных переменных).
        \item Применить математически корректное кодирование качественных категориальных признаков (бинаризация дихотомических признаков, порядковое кодирование ранговых категорий, One-Hot кодирование номинальных факторов).
    \end{enumerate}
    \item \textbf{Масштабирование данных:} выполнить стандартизацию и/или нормализацию числовых признаков; провести сопоставительный анализ распределений до и после масштабирования.
\end{enumerate}

\subsection{Спецификация исследуемого датасета}

В качестве объекта исследования выбран массив данных \textbf{«Student Performance Factors»} (\texttt{StudentPerformanceFactors.csv}), содержащий комплексные академические, психологические, бытовые и социально-демографические параметры 6607 студентов, а также их итоговые результаты на экзамене (\texttt{Exam\_Score}). Исходная матрица данных имеет размерность: \textbf{6607 строк и 20 столбцов} (19 признаков-предикторов и 1 целевая переменная). Подробная спецификация признаков приведена в таблице~\ref{tab:features_spec}.

\begin{table}[H]
    \centering
    \caption{Спецификация признаков датасета Student Performance Factors}
    \label{tab:features_spec}
    \fontsize{8.5pt}{10.5pt}\selectfont
    \begin{tabularx}{\textwidth}{|c|>{\raggedright\arraybackslash}p{3.8cm}|c|c|>{\raggedright\arraybackslash}X|}
        \hline
        \textbf{№} & \textbf{Идентификатор} & \textbf{Тип данных} & \textbf{Роль} & \textbf{Предметно-содержательное описание} \\
        \hline
        1 & \texttt{Hours\_Studied} & \texttt{int64} & Фактор & Часов самостоятельной учебы в неделю \\
        \hline
        2 & \texttt{Attendance} & \texttt{int64} & Фактор & Процент посещаемости учебных занятий (0--100\%) \\
        \hline
        3 & \texttt{Parental\_Involvement} & \texttt{object} (str) & Фактор & Вовлеченность родителей ('Low', 'Medium', 'High') \\
        \hline
        4 & \texttt{Access\_to\_Resources} & \texttt{object} (str) & Фактор & Доступность ресурсов ('Low', 'Medium', 'High') \\
        \hline
        5 & \texttt{Extracurricular\_Activities} & \texttt{object} (str) & Фактор & Внеучебная активность ('No', 'Yes') \\
        \hline
        6 & \texttt{Sleep\_Hours} & \texttt{int64} & Фактор & Среднее количество часов сна в сутки (4--10) \\
        \hline
        7 & \texttt{Previous\_Scores} & \texttt{int64} & Фактор & Результаты прошлых тестирований (50--100) \\
        \hline
        8 & \texttt{Motivation\_Level} & \texttt{object} (str) & Фактор & Уровень мотивации ('Low', 'Medium', 'High') \\
        \hline
        9 & \texttt{Internet\_Access} & \texttt{object} (str) & Фактор & Наличие интернета дома ('No', 'Yes') \\
        \hline
        10 & \texttt{Tutoring\_Sessions} & \texttt{int64} & Фактор & Число занятий с репетитором в месяц (0--8) \\
        \hline
        11 & \texttt{Family\_Income} & \texttt{object} (str) & Фактор & Доход семьи ('Low', 'Medium', 'High') \\
        \hline
        12 & \texttt{Teacher\_Quality} & \texttt{object} (str) & Фактор & Оценка квалификации учителей ('Low', 'Med', 'High') \\
        \hline
        13 & \texttt{School\_Type} & \texttt{object} (str) & Фактор & Тип учебного заведения ('Public', 'Private') \\
        \hline
        14 & \texttt{Peer\_Influence} & \texttt{object} (str) & Фактор & Влияние окружения ('Positive', 'Neutral', 'Negative') \\
        \hline
        15 & \texttt{Physical\_Activity} & \texttt{int64} & Фактор & Часов спорта в неделю (0--6) \\
        \hline
        16 & \texttt{Learning\_Disabilities} & \texttt{object} (str) & Фактор & Трудности в обучении ('No', 'Yes') \\
        \hline
        17 & \texttt{Parental\_Education\_Level} & \texttt{object} (str) & Фактор & Образование родителей ('High School', 'College', ...) \\
        \hline
        18 & \texttt{Distance\_from\_Home} & \texttt{object} (str) & Фактор & Удаленность от школы ('Near', 'Moderate', 'Far') \\
        \hline
        19 & \texttt{Gender} & \texttt{object} (str) & Фактор & Пол обучающегося ('Male', 'Female') \\
        \hline
        20 & \texttt{Exam\_Score} & \texttt{int64} & \textbf{Целевой} & Итоговый балл студента на экзамене (0--100) \\
        \hline
    \end{tabularx}
\end{table}

\subsection{Матрица верификации выполнения требований задания}

В таблице~\ref{tab:compliance_matrix} представлено прямое сопоставление нормативных пунктов задания кафедры с фактически реализованными модулями программного комплекса.

\begin{table}[H]
    \centering
    \caption{Матрица верификации соответствия работы официальному заданию}
    \label{tab:compliance_matrix}
    \fontsize{8.5pt}{10.5pt}\selectfont
    \begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{2.8cm}|>{\raggedright\arraybackslash}p{4.2cm}|>{\raggedright\arraybackslash}X|>{\centering\arraybackslash}p{2.0cm}|}
        \hline
        \textbf{Пункт задания} & \textbf{Нормативное требование} & \textbf{Фактическая реализация в проекте} & \textbf{Статус} \\
        \hline
        1. Импорт данных & Загрузка табличного массива & Модуль \texttt{data\_loader.py}, функция \texttt{load\_dataset()}, 6607 строк, 20 столбцов & Выполнено \\
        \hline
        2. Анализ пропусков & Статистический и визуальный поиск \texttt{NaN} & \texttt{get\_missing\_summary()}, Рис.~\ref{fig:miss_before} (Heatmap) и Рис.~\ref{fig:miss_bar} (Bar chart) & Выполнено \\
        \hline
        2. Заполнение пропусков & Устранение выявленных пропусков & \texttt{impute\_missing\_with\_mode()}: категориальные признаки заполнены модой & Выполнено \\
        \hline
        2. Контроль пропусков & Графическая верификация чистоты данных & Повторный Heatmap (Рис.~\ref{fig:miss_after}) с бейджем «100\% заполненность» (0 пропусков) & Выполнено \\
        \hline
        2. Анализ выбросов & Расчет статистик и графики распределений & Расчет границ $1.5 \times IQR$, Рис.~\ref{fig:dist_before} (Hist+KDE), Рис.~\ref{fig:box_before} (BoxPlot) & Выполнено \\
        \hline
        2. Устранение выбросов & Ликвидация аномалий и артефактов & Коррекция ошибки 101 балл, винсоризация (\texttt{clip}) по границам $IQR$ & Выполнено \\
        \hline
        2. Контроль выбросов & Графическая проверка отсутствия аномалий & Контрольные BoxPlot (Рис.~\ref{fig:box_after}), гистограммы (Рис.~\ref{fig:dist_after}), матрица корреляций (Рис.~\ref{fig:corr}) & Выполнено \\
        \hline
        3. Feature Engineering & Генерация $\ge 2$ новых признаков & Синтезировано \textbf{5 предметных признаков} в \texttt{feature\_engineering.py}, Рис.~\ref{fig:fe} & Выполнено \\
        \hline
        3. Кодирование категорий & OHE, бинаризация и другие методы & Бинаризация 5 колонок, Ordinal 7 колонок, One-Hot 1 колонки (итого 27 признаков) & Выполнено \\
        \hline
        4. Масштабирование данных & Стандартизация и/или нормализация & Реализованы \texttt{MinMaxScaler} и \texttt{StandardScaler}, Рис.~\ref{fig:scaling} (сравнение KDE) & Выполнено \\
        \hline
    \end{tabularx}
\end{table}

% ==============================================================================
% 4 ВЕСЬ КОД РЕШЕНИЯ ЛАБОРАТОРНОЙ РАБОТЫ
% ==============================================================================
\clearpage
\section{Весь код решения лабораторной работы}

\subsection{Исходный код модуля конфигурации (файл config.py)}
\begin{CodeListing}
'''

code_sections = [
    (config_code, r'''\end{CodeListing}

\clearpage
\subsection{Исходный код модуля загрузки данных (файл data\_loader.py)}
\begin{CodeListing}
'''),
    (data_loader_code, r'''\end{CodeListing}

\clearpage
\subsection{Исходный код модуля очистки данных и обработки выбросов (файл preprocessor.py)}
\begin{CodeListing}
'''),
    (preprocessor_code, r'''\end{CodeListing}

\clearpage
\subsection{Исходный код модуля генерации признаков и кодирования (файл feature\_engineering.py)}
\begin{CodeListing}
'''),
    (fe_code, r'''\end{CodeListing}

\clearpage
\subsection{Исходный код модуля масштабирования (файл scaler.py)}
\begin{CodeListing}
'''),
    (scaler_code, r'''\end{CodeListing}

\clearpage
\subsection{Исходный код модуля расширенной визуализации (файл visualizer.py)}
\begin{CodeListing}
'''),
    (visualizer_code, r'''\end{CodeListing}

\clearpage
\subsection{Исходный код главного модуля оркестрации пайплайна (файл main.py)}
\begin{CodeListing}
'''),
    (main_code, r'''\end{CodeListing}

\clearpage
\subsection{Исходный код точки входа (файл solution.py)}
\begin{CodeListing}
'''),
    (solution_code, r'''\end{CodeListing}
''')
]

results_section = r'''
% ==============================================================================
% 5 ЭКСПЕРИМЕНТАЛЬНЫЕ РЕЗУЛЬТАТЫ И ГРАФИЧЕСКИЙ АНАЛИЗ
% ==============================================================================
\clearpage
\section{Экспериментальные результаты и графический анализ}

\subsection{Первичный структурный и статистический анализ выборки}

В ходе импорта датасета функцией \texttt{load\_dataset()} сформирован объект \texttt{pandas.DataFrame} размерностью 6607 строк и 20 столбцов. Функция \texttt{get\_column\_types()} классифицировала признаки выборки на 7 числовых (\texttt{int64}) и 13 категориальных строковых (\texttt{object}).

В таблице~\ref{tab:initial_numeric_stats} представлены результаты вычисления параметров описательной статистики числовых признаков исходной выборки.

\begin{table}[H]
    \centering
    \caption{Описательные статистики числовых признаков исходного датасета}
    \label{tab:initial_numeric_stats}
    \fontsize{8.5pt}{10.5pt}\selectfont
    \begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{3.5cm}|c|c|c|c|c|c|c|c|}
        \hline
        \textbf{Признак} & \textbf{Count} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Q1} & \textbf{Med} & \textbf{Q3} & \textbf{Max} \\
        \hline
        \texttt{Hours\_Studied} & 6607 & 19.98 & 5.99 & 1.00 & 16.00 & 20.00 & 24.00 & 44.00 \\
        \hline
        \texttt{Attendance} & 6607 & 79.98 & 11.55 & 60.00 & 70.00 & 80.00 & 90.00 & 100.00 \\
        \hline
        \texttt{Sleep\_Hours} & 6607 & 7.03 & 1.47 & 4.00 & 6.00 & 7.00 & 8.00 & 10.00 \\
        \hline
        \texttt{Previous\_Scores} & 6607 & 75.07 & 14.40 & 50.00 & 63.00 & 75.00 & 88.00 & 100.00 \\
        \hline
        \texttt{Tutoring\_Sessions} & 6607 & 1.49 & 1.23 & 0.00 & 1.00 & 1.00 & 2.00 & 8.00 \\
        \hline
        \texttt{Physical\_Activity} & 6607 & 2.97 & 1.03 & 0.00 & 2.00 & 3.00 & 4.00 & 6.00 \\
        \hline
        \texttt{Exam\_Score} & 6607 & 67.24 & 3.89 & 55.00 & 65.00 & 67.00 & 69.00 & \textbf{101.00} \\
        \hline
    \end{tabularx}
\end{table}

\textbf{Аналитические выводы первичного осмотра:}
\begin{enumerate}
    \item В целевой переменной \texttt{Exam\_Score} зафиксирована грубая логическая аномалия: значение $\max = 101.0$, превышающее теоретический 100-балльный максимум экзаменационной шкалы.
    \item Средний балл на экзамене составляет $67.24$ при стандартном отклонении $3.89$. Совпадение медианы ($67.0$) и среднего ($67.24$) свидетельствует о симметричности ядра распределения.
    \item Посещаемость варьируется от 60\% до 100\% (среднее 79.98\%), сон --- от 4 до 10 часов в сутки (среднее 7.03 ч).
\end{enumerate}

\subsection{Анализ и устранение пропущенных значений}

Статистический анализ функцией \texttt{get\_missing\_summary()} выявил наличие пропусков строго в трех категориальных столбцах при полном отсутствии пропусков в числовых атрибутах (таблица~\ref{tab:missing_summary}).

\begin{table}[H]
    \centering
    \caption{Сводка обнаруженных пропусков по категориальным признакам}
    \label{tab:missing_summary}
    \fontsize{8.5pt}{10.5pt}\selectfont
    \begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{5.5cm}|>{\centering\arraybackslash}p{3.2cm}|>{\centering\arraybackslash}p{3.2cm}|>{\raggedright\arraybackslash}X|}
        \hline
        \textbf{Наименование признака} & \textbf{Пропусков (шт.)} & \textbf{Доля от выборки (\%)} & \textbf{Модальное значение ($M_0$)} \\
        \hline
        \texttt{Parental\_Education\_Level} & 90 & 1.36\% & \texttt{'High School'} \\
        \hline
        \texttt{Teacher\_Quality} & 78 & 1.18\% & \texttt{'Medium'} \\
        \hline
        \texttt{Distance\_from\_Home} & 67 & 1.01\% & \texttt{'Near'} \\
        \hline
    \end{tabularx}
\end{table}

Суммарно пропущено 235 ячеек (менее 3.5\% от общего объема строк). Применение удаления наблюдений привело бы к необоснованной потере полезной информации. Выполнена импутация модой посредством функции \texttt{impute\_missing\_with\_mode()}.

На рисунках~\ref{fig:miss_before} и~\ref{fig:miss_bar} представлена визуализация структуры пропусков до очистки. На рисунке~\ref{fig:miss_after} приведен контрольный тепловой снимок матрицы после импутации модой, подтверждающий достижение 100\% заполненности.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\linewidth, keepaspectratio]{images/01_missing_values_before_heatmap.png}
    \caption{Тепловая карта пропущенных значений в исходном массиве данных (ДО заполнения)}
    \label{fig:miss_before}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\linewidth, keepaspectratio]{images/02_missing_values_bar.png}
    \caption{Количество и процентная доля пропущенных значений по категориальным признакам}
    \label{fig:miss_bar}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\linewidth, keepaspectratio]{images/03_missing_values_after_heatmap.png}
    \caption{Контрольная тепловая карта после импутации модой (0 пропусков во всем датасете)}
    \label{fig:miss_after}
\end{figure}

\subsection{Анализ и устранение аномальных значений и выбросов}

С использованием формулы межквартильного размаха Тьюки $IQR = Q_3 - Q_1$ рассчитаны границы выбросов $[Q_1 - 1.5 \times IQR, \; Q_3 + 1.5 \times IQR]$. Результаты представлены в таблице~\ref{tab:iqr_analysis}.

\begin{table}[H]
    \centering
    \caption{Статистика обнаружения выбросов по правилу $1.5 \times IQR$}
    \label{tab:iqr_analysis}
    \fontsize{8.5pt}{10.5pt}\selectfont
    \begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{3.4cm}|c|c|c|c|c|c|c|c|}
        \hline
        \textbf{Признак} & \textbf{Q1} & \textbf{Q3} & \textbf{IQR} & \textbf{Нижн.} & \textbf{Верхн.} & \textbf{Мин/Макс} & \textbf{Выбросов} & \textbf{\%} \\
        \hline
        \texttt{Hours\_Studied} & 16.0 & 24.0 & 8.0 & 4.0 & 36.0 & 1 / 44 & 43 & 0.65\% \\
        \hline
        \texttt{Attendance} & 70.0 & 90.0 & 20.0 & 40.0 & 120.0 & 60 / 100 & 0 & 0.00\% \\
        \hline
        \texttt{Sleep\_Hours} & 6.0 & 8.0 & 2.0 & 3.0 & 11.0 & 4 / 10 & 0 & 0.00\% \\
        \hline
        \texttt{Previous\_Scores} & 63.0 & 88.0 & 25.0 & 25.5 & 125.5 & 50 / 100 & 0 & 0.00\% \\
        \hline
        \texttt{Tutoring\_Sessions} & 1.0 & 2.0 & 1.0 & -0.5 $\to$ 0 & 3.5 & 0 / 8 & 430 & 6.51\% \\
        \hline
        \texttt{Physical\_Activity} & 2.0 & 4.0 & 2.0 & -1.0 $\to$ 0 & 7.0 & 0 / 6 & 0 & 0.00\% \\
        \hline
        \texttt{Exam\_Score} & 65.0 & 69.0 & 4.0 & 59.0 & 75.0 & 55 / 101 & 104 & 1.57\% \\
        \hline
    \end{tabularx}
\end{table}

На рисунках~\ref{fig:dist_before} и~\ref{fig:box_before} проиллюстрированы распределения до фильтрации: четко видны изолированные точки за пределами усов в переменных \texttt{Hours\_Studied}, \texttt{Exam\_Score} и \texttt{Tutoring\_Sessions}, а также артефактная точка $101$ на диаграмме размаха.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\linewidth, keepaspectratio]{images/04_distributions_before.png}
    \caption{Гистограммы и оценки ядерной плотности (KDE) признаков (ДО устранения выбросов)}
    \label{fig:dist_before}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\linewidth, keepaspectratio]{images/05_boxplots_before.png}
    \caption{Диаграммы размаха BoxPlot (ДО устранения выбросов: стрелка к аномалии 101)}
    \label{fig:box_before}
\end{figure}

Процедура устранения аномалий включала два шага:
\begin{enumerate}
    \item Коррекция логической ошибки: запись \texttt{Exam\_Score} $= 101$ приведена к 100.
    \item Винсоризация (\texttt{clipping}): значения \texttt{Hours\_Studied} ограничены диапазоном $[4.0, 36.0]$; значения \texttt{Exam\_Score} ограничены диапазоном $[59.0, 75.0]$; значения \texttt{Tutoring\_Sessions} ограничены диапазоном $[0.0, 3.5]$.
\end{enumerate}

На контрольных рисунках~\ref{fig:box_after} и~\ref{fig:dist_after} подтверждено полное отсутствие вырожденных точек за усами и стабилизация формы плотностей распределений.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\linewidth, keepaspectratio]{images/06_boxplots_after.png}
    \caption{Контрольные диаграммы размаха BoxPlot (ПОСЛЕ винсоризации выбросов)}
    \label{fig:box_after}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\linewidth, keepaspectratio]{images/07_distributions_after.png}
    \caption{Гистограммы и KDE числовых признаков (ПОСЛЕ устранения выбросов)}
    \label{fig:dist_after}
\end{figure}

\subsection{Корреляционный анализ факторов академической успеваемости}

На рисунке~\ref{fig:corr} представлена нижнетреугольная тепловая карта матрицы парных коэффициентов корреляции Пирсона, отсортированная по силе взаимосвязи с целевой переменной \texttt{Exam\_Score}.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth, keepaspectratio]{images/08_correlation_matrix.png}
    \caption{Матрица корреляций Пирсона (отсортирована по силе связи с Exam\_Score)}
    \label{fig:corr}
\end{figure}

\textbf{Выводы корреляционного анализа:}
\begin{itemize}
    \item Ключевыми факторами успеха на экзамене выступают посещаемость (\texttt{Attendance}, $r = 0.58$) и время самостоятельной учебной работы (\texttt{Hours\_Studied}, $r = 0.45$).
    \item Базовый уровень подготовки (\texttt{Previous\_Scores}) вносит умеренный позитивный вклад ($r = 0.18$).
    \item Часы сна (\texttt{Sleep\_Hours}) и спорт (\texttt{Physical\_Activity}) имеют слабую линейную связь с экзаменом ($|r| < 0.05$), что обосновывает конструирование нелинейных композитных признаков.
\end{itemize}

\subsection{Синтез новых признаков (Feature Engineering)}

На основе предметных зависимостей сконструировано 5 новых признаков:
\begin{enumerate}
    \item \texttt{Study\_Sleep\_Ratio} $= \frac{\text{Hours\_Studied}}{\text{Sleep\_Hours}}$ --- баланс интеллектуального труда и физиологического восстановления.
    \item \texttt{Total\_Study\_Effort} $= \text{Hours\_Studied} + 2.0 \times \text{Tutoring\_Sessions} + 0.1 \times \text{Attendance}$ --- комплексный аддитивный индекс академических усилий.
    \item \texttt{Score\_Improvement} $= \text{Exam\_Score} - \text{Previous\_Scores}$ --- академический прирост знаний.
    \item \texttt{Physical\_Study\_Balance} $= \frac{\text{Physical\_Activity}}{\text{Hours\_Studied} + 1}$ --- гармоничность умственной и физической активности.
    \item \texttt{High\_Attendance\_Flag} $= \mathbb{I}(\text{Attendance} \ge 85\%)$ --- предикат высокой дисциплины.
\end{enumerate}

На рисунке~\ref{fig:fe} приведен детальный аналитический профиль сгенерированных переменных. Анализ показал, что студенты с флагом высокой посещаемости ($\ge 85\%$) сдали экзамен в среднем на \textbf{69.8 балла}, тогда как студенты группы низкой посещаемости --- всего на \textbf{64.7 балла}, что подтверждает высокую информационную ценность синтезированного признака.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\linewidth, keepaspectratio]{images/09_feature_engineering.png}
    \caption{Аналитический профиль сгенерированных признаков (Feature Engineering)}
    \label{fig:fe}
\end{figure}

\subsection{Результаты кодирования категориальных признаков}

Выполнено преобразование всех качественных переменных:
\begin{itemize}
    \item \textbf{Бинаризация:} 5 дихотомических признаков (\texttt{Extracurricular\_Activities}, \texttt{Internet\_Access}, \texttt{Learning\_Disabilities}, \texttt{School\_Type}, \texttt{Gender}) переведены в числовые значения 0 и 1.
    \item \textbf{Порядковое кодирование (Ordinal):} 7 ранговых шкал (\texttt{Parental\_Involvement}, \texttt{Access\_to\_Resources}, \texttt{Motivation\_Level}, \texttt{Family\_Income}, \texttt{Teacher\_Quality}, \texttt{Distance\_from\_Home}, \texttt{Parental\_Education\_Level}) монотонно отображены в целочисленные ранги 0, 1, 2.
    \item \textbf{One-Hot Encoding (OHE):} номинальный признак \texttt{Peer\_Influence} преобразован в 3 бинарных столбца (\texttt{Peer\_Influence\_Negative}, \texttt{Peer\_Influence\_Neutral}, \texttt{Peer\_Influence\_Positive}).
\end{itemize}

Итоговая матрица признакового пространства расширена с 20 до \textbf{27 столбцов} при полном сохранении 6607 строк.

\subsection{Сравнительный анализ стратегий масштабирования данных}

Выполнено масштабирование 11 непрерывных признаков выборки алгоритмами \texttt{MinMaxScaler} и \texttt{StandardScaler}. В таблице~\ref{tab:scaling_stats} приведены сравнительные статистики для репрезентативных колонок.

\begin{table}[H]
    \centering
    \caption{Сравнительный анализ числовых характеристик до и после масштабирования}
    \label{tab:scaling_stats}
    \fontsize{8.5pt}{10.5pt}\selectfont
    \begin{tabularx}{\textwidth}{|>{\raggedright\arraybackslash}p{3.5cm}|>{\raggedright\arraybackslash}p{3.2cm}|c|c|c|}
        \hline
        \textbf{Признак} & \textbf{Метрика} & \textbf{Оригинал} & \textbf{MinMaxScaler} & \textbf{StandardScaler} \\
        \hline
        \texttt{Hours\_Studied} & Среднее (Mean) & 19.9718 & 0.4991 & -0.0000 \\
        & Станд. откл. (Std) & 5.9510 & 0.1860 & 1.0001 \\
        & Минимум (Min) & 4.0000 & 0.0000 & -2.6841 \\
        & Максимум (Max) & 36.0000 & 1.0000 & 2.6936 \\
        \hline
        \texttt{Attendance} & Среднее (Mean) & 79.9774 & 0.4994 & -0.0000 \\
        & Станд. откл. (Std) & 11.5475 & 0.2887 & 1.0001 \\
        & Минимум (Min) & 60.0000 & 0.0000 & -1.7302 \\
        & Максимум (Max) & 100.0000 & 1.0000 & 1.7341 \\
        \hline
        \texttt{Exam\_Score} & Среднее (Mean) & 67.1271 & 0.5079 & -0.0000 \\
        & Станд. откл. (Std) & 3.3729 & 0.2108 & 1.0001 \\
        & Минимум (Min) & 59.0000 & 0.0000 & -2.4097 \\
        & Максимум (Max) & 75.0000 & 1.0000 & 2.3343 \\
        \hline
        \texttt{Total\_Study\_Effort} & Среднее (Mean) & 30.8423 & 0.5047 & 0.0000 \\
        & Станд. откл. (Std) & 6.4088 & 0.1552 & 1.0001 \\
        & Минимум (Min) & 10.0000 & 0.0000 & -3.2524 \\
        & Максимум (Max) & 51.3000 & 1.0000 & 3.1923 \\
        \hline
    \end{tabularx}
\end{table}

На рисунке~\ref{fig:scaling} представлена сравнительная матрица $3 \times 4$ оценок ядерной плотности (KDE) для четырех признаков при трех стратегиях масштабирования.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\linewidth, keepaspectratio]{images/10_scaling_comparison.png}
    \caption{Сравнительный анализ профилей плотности (KDE): Исходные vs MinMax vs Standard}
    \label{fig:scaling}
\end{figure}

\textbf{Главный вывод по масштабированию:} Проведенный анализ эмпирически подтверждает доказанную теорему: и \texttt{MinMaxScaler}, и \texttt{StandardScaler} являются строго аффинными (линейными) преобразованиями. Они приводят шкалы к нормализованным диапазонам ($[0, 1]$ и $\mu=0, \sigma=1$ соответственно), абсолютно не искажая внутреннюю топологию, асимметрию и относительное расположение квантилей исходных плотностей распределения.

% ==============================================================================
% 6 ВЫВОДЫ ПО ЛАБОРАТОРНОЙ РАБОТЕ
% ==============================================================================
\clearpage
\section{Выводы по лабораторной работе}

В ходе выполнения лабораторной работы № 2 по дисциплине «Методы машинного обучения» были достигнуты следующие академические и прикладные результаты:

\begin{enumerate}
    \item Разработан законченный модульный программный конвейер предварительной обработки структурированных данных на языке Python 3.12 с использованием современных библиотек машинного обучения (\texttt{pandas}, \texttt{numpy}, \texttt{scikit-learn}, \texttt{seaborn}, \texttt{matplotlib}).
    \item На основе статистического анализа и графической визуализации тепловыми картами выявлены скрытые пропуски в 3 категориальных атрибутах. Научно обоснована и реализована стратегия импутации модой, сохранившая 100\% объема обучающей выборки (все 6607 наблюдений) и подтвержденная контрольной тепловой картой с нулевым остаточным числом пропусков.
    \item С использованием статистического правила Тьюки ($1.5 \times IQR$) и диаграмм BoxPlot обнаружены и устранены грубая логическая аномалия в целевой переменной (\texttt{Exam\_Score} $= 101 \to 100$), а также экстремальные выбросы в переменных \texttt{Hours\_Studied}, \texttt{Exam\_Score} и \texttt{Tutoring\_Sessions}. Доказана эффективность винсоризации (\textit{clipping}), позволившей исключить тяжелые дестабилизирующие хвосты распределений без удаления наблюдений.
    \item Проведен корреляционный анализ, выявивший ведущие линейные детерминанты успеваемости студентов: посещаемость занятий ($r = 0.58$) и количество часов самостоятельной учебы ($r = 0.45$).
    \item Выполнена инженерия признаков (Feature Engineering): сконструировано 5 предметных показателей (\texttt{Study\_Sleep\_Ratio}, \texttt{Total\_Study\_Effort}, \texttt{Score\_Improvement}, \texttt{Physical\_Study\_Balance}, \texttt{High\_Attendance\_Flag}). Доказана аналитическая значимость флага дисциплины \texttt{High\_Attendance\_Flag}, показавшего разрыв в среднем балле экзамена в 5.1 единицы между группами.
    \item Реализованы корректные схемы оцифровки категориальных данных: бинаризация 5 колонок, порядковое кодирование (Ordinal Encoding) 7 колонок и One-Hot Encoding номинального признака окружения сверстников, что исключило возникновение ложного метрического порядка и расширило размерность выборки до 27 параметров.
    \item Выполнено масштабирование признакового пространства методами \texttt{MinMaxScaler} и \texttt{StandardScaler}. Теоретически и экспериментально доказана инвариантность формы ядерной оценки плотности (KDE) при линейных преобразованиях масштаба.
    \item Все требования рабочей программы лабораторной работы выполнены в полном объеме. Сформированы 4 обработанных набора данных в формате CSV и 10 графических материалов высокого разрешения (300 DPI).
\end{enumerate}

\end{document}
'''

import zipfile

with open(OUT_TEX_PATH, 'w', encoding='utf-8') as out_f:
    out_f.write(header)
    for code_chunk, separator in code_sections:
        out_f.write(code_chunk)
        out_f.write(separator)
    out_f.write(results_section)

# Также сохраняем копию в корне каталога Отчет
root_report_tex = os.path.join(REPORT_DIR, 'report.tex')
with open(root_report_tex, 'w', encoding='utf-8') as f:
    with open(OUT_TEX_PATH, 'r', encoding='utf-8') as src:
        f.write(src.read())

print(f'Successfully generated {OUT_TEX_PATH} ({os.path.getsize(OUT_TEX_PATH)} bytes)')
print(f'Successfully generated {root_report_tex} ({os.path.getsize(root_report_tex)} bytes)')

# Собираем ZIP-архивы для Overleaf в обоих каталогах
zip_paths = [
    os.path.join(REPORT_DIR, 'Отчет_Overleaf.zip'),
    os.path.join(OVERLEAF_DIR, 'Отчет_Overleaf.zip')
]

for zp in zip_paths:
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_TEX_PATH, 'report.tex')
        img_dir = os.path.join(OVERLEAF_DIR, 'images')
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, OVERLEAF_DIR).replace('\\', '/')
                z.write(full_p, rel_p)
    print(f'Successfully packed {zp} ({os.path.getsize(zp)} bytes)')

