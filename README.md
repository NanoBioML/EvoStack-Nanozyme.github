# NanozymeDesigner
Physics-Informed Multi-Output Machine Learning for Nanozyme Design

![Static Badge](https://img.shields.io/badge/Zhakupov-NanozymeDesigner-blue)
![GitHub top language](https://img.shields.io/github/languages/top/DanielZhakupov/Physics-Informed-Nanozyme)
![GitHub](https://img.shields.io/github/license/DanielZhakupov/Physics-Informed-Nanozyme)
![DOI](https://img.shields.io/badge/DOI-10.26434/chemrxiv.15000742/v1-blue)

![Logotype](./docs/header.jpg) 

## Описание
Программное обеспечение для рационального дизайна нанозимов на стыке биоинженерии и машинного обучения. Система предсказывает кинетические параметры (Km, kcat, Vmax), используя физически-информированные дескрипторы и ансамбли деревьев (ExtraTreesRegressor).

## Установка и запуск (Universal)
Для работы проекта требуется Python 3.11 или выше. Выполните следующие команды последовательно:

1. git clone https://github.com/DanielZhakupov/Physics-Informed-Nanozyme.git
2. cd Physics-Informed-Nanozyme
3. python -m venv venv
4. source venv/bin/activate  # Для Windows: venv\Scripts\activate
5. pip install -r requirements.txt
6. python designer.py

## Документация (docs/ru/)
Вся пользовательская документация и научные основы проекта находятся в папке docs.
- [Введение в проект](./docs/ru/intro.md): описание PIML-фреймворка и целей исследования.
- [Основы Nano-ML](./docs/ru/ml-basics.md): почему ExtraTreesRegressor лучше нейросетей на малых данных (n=86) и как работают физические дескрипторы (Km, Arrhenius factor).

## Поддержка и зависимости
По всем вопросам создавайте Issue в репозитории. Проект зависит от библиотек: Scikit-learn, Pandas, NumPy, RDKit.

## Описание коммитов
| Название | Описание |
|----------|----------|
| build | Сборка проекта или изменения внешних зависимостей |
| science | Изменения в физических константах или термодинамике |
| feat | Добавление нового функционала (модели, дескрипторы) |
| fix | Исправление ошибок в расчетах или коде |
| docs | Обновление документации или README |
| perf | Оптимизация производительности |
| style | Правки по кодстайлу (PEP8) |

## Примеры использования Git
- git commit -m "science: update activation energy constants for POD"
- git commit -m "feat: implement substrate SMILES embedding for H2O2"
- git commit -m "fix: correct Vmax clipping logic in RegressorChain"

## Лицензия
Проект распространяется под лицензией MIT.
