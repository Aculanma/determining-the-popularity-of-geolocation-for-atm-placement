# Determining-the-Popularity-of-Geolocation-for-ATM-Placement  
 
| __Тема проекта__ | Определение популярности геолокации для размещения банкомата |
|:----|:----|
| __Куратор__ | Александра Ковалева (@ak0va) |
| __Участники__ | 1. Николай Цуканов — @ntsukanov, [ntsukanov](https://github.com/ntsukanov)<br> 2. Ильдус Исхаков — @Aculanma, [Aculanma](https://github.com/Aculanma/Aculanma)<br> 3. Кирилл Устименко — @diefrein, [diefrein](https://github.com/diefrein)<br> 4. Анастасия Волокитина — @volokitinaa, [volokitinaa](https://github.com/volokitinaa)<br> |
| __Описание проекта__ | В рамках данного проекта предстоит построить модель, которая по географическим координатам и адресу точки выдаст оценку индекса популярности банкомата в этой самой локации | 

Инструкция по установке и развертыванию сервиса:

1. Установить Git на сервер
Инструкция по установке: https://git-scm.com/book/ru/v2/Введение-Установка-Git

2. Установить docker на сервер
Инструкция по установке: https://docs.docker.com/engine/install/ubuntu/

3. Склонировать репозиторий с сервисом
git clone https://github.com/Aculanma/determining-the-popularity-of-geolocation-for-atm-placement.git

4. Переходим в корневую папку проекта 
cd determining-the-popularity-of-geolocation-for-atm-placement

5. Запускаем сервисы в docker-контейнерах
docker compose up -d
По умолчанию streamlit запускается на localhost:8501