# Image Batch

Application de bureau locale (Mac, Windows, Linux) pour traiter des images par lot :
recadrer, changer le ratio en ajoutant du fond, redimensionner, convertir et retoucher.
Tout se passe sur votre ordinateur, rien n'est envoyé en ligne.

Stack : Python 3.10+, PySide6 (interface Qt native) et Pillow (traitement d'image).

## Lancer l'application (Mac)

1. Ouvrez le Terminal et clonez le dépôt :
   ```bash
   git clone https://github.com/martinsilvestre-ams/batch-cooking.git
   cd batch-cooking
   ```
2. Lancez `./lancer.command` (ou double-cliquez dessus dans le Finder).
   Le premier lancement installe les dépendances dans `.venv` (1 à 2 minutes).

Python 3 est requis : `python3 --version` doit répondre. Sinon : `brew install python`.

### Créer une vraie application `.app`

```bash
./build_mac.sh
```
Cela produit `dist/Image Batch.app`, à glisser dans le dossier Applications.
Au premier lancement, macOS peut bloquer l'app (non signée) : clic droit sur l'app, puis « Ouvrir ».

## Fonctionnalités

- **Sélection** : fichiers multiples, dossier entier (avec ou sans sous-dossiers), ou glisser-déposer.
  Formats lus : JPEG, PNG, WEBP, TIFF, BMP, GIF, et HEIC (photos iPhone).
- **Recadrage** : centré selon un ratio (1:1, 4:3, 3:2, 16:9, 4:5, 2:3, 9:16), ou zone dessinée
  à la souris sur l'aperçu, appliquée à toutes les images (en proportions).
- **Changer le ratio en ajoutant du fond** : l'image entière est conservée et la toile est agrandie
  au ratio voulu. Fond au choix : image floutée, couleur unie, bords étirés, ou transparent.
- **Redimensionnement** : largeur fixe, hauteur fixe, tenir dans un cadre, dimensions exactes,
  pourcentage, avec option « ne jamais agrandir ».
- **Retouches** : rotation, miroirs, luminosité, contraste, saturation, netteté, noir et blanc.
- **Export** : JPEG, PNG, WEBP, TIFF, BMP ou format d'origine, qualité réglable, suffixe de nom,
  conservation optionnelle de l'EXIF et du profil couleur. Les originaux ne sont jamais écrasés.

Ordre d'application : recadrage → rotation/miroir → retouches → ajout de fond → redimensionnement.

## Développement

```bash
pip install -r requirements.txt pytest
python -m pytest tests
python main.py
```

- `imagebatch/processing.py` : tout le traitement (sans interface, testable).
- `imagebatch/app.py` : l'interface.
