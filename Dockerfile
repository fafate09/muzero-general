# Utilisation d'une image de base qui fonctionne bien sous Windows
FROM mcr.microsoft.com/windows/servercore:ltsc2019

# Installer Python et pip
RUN powershell -Command \
    Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.9.6/python-3.9.6.exe -OutFile python-installer.exe; \
    Start-Process -Wait -FilePath python-installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1'; \
    Remove-Item -Force python-installer.exe

# Mettre à jour pip et installer virtualenv
RUN python -m pip install --upgrade pip
RUN pip install virtualenv

# Créer un environnement virtuel
RUN python -m venv venv

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers locaux dans le conteneur
COPY . /app

# Installer les dépendances du fichier requirements.txt
RUN /app/venv/Scripts/pip install --upgrade pip
RUN /app/venv/Scripts/pip install -r requirements.txt

# Commande pour exécuter votre application
CMD ["python", "votre_application.py"]
