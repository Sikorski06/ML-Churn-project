# Bierzemy lekką, oficjalną wersję Pythona (baza naszego kontenera)
FROM python:3.11-slim

# Ustawiamy katalog roboczy wewnątrz kontenera na /app
WORKDIR /app

# Kopiujemy plik z listą bibliotek
COPY requirements.txt .

# Instalujemy biblioteki (flaga --no-cache-dir zmniejsza wagę obrazu)
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy całą resztę projektu do kontenera
COPY . .

# Informujemy Dockera, że aplikacja będzie działać na porcie 8000
EXPOSE 8000

# Komenda, która uruchomi się, gdy włączymy kontener
# (Ważne: --host 0.0.0.0 pozwala na dostęp z zewnątrz kontenera)
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]