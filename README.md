# Numbered - Batch Cooking

Application de bureau locale (Mac, Windows, Linux) pour traiter des images par lot :
recadrer, changer le ratio en ajoutant du fond, redimensionner, convertir et retoucher.
Tout se passe sur votre ordinateur, rien n'est envoyé en ligne.

Stack : Python 3.9+ (le Python fourni par macOS suffit), PySide6 (interface Qt native) et Pillow (traitement d'image).

## Lancer l'application (Mac)

1. Ouvrez le Terminal et clonez le dépôt :
   ```bash
   git clone https://github.com/martinsilvestre-ams/batch-cooking.git
   cd batch-cooking
   ```
2. Lancez `./lancer.command` (ou double-cliquez dessus dans le Finder).
   Le premier lancement installe les dépendances dans `.venv` (1 à 2 minutes).

Python 3.9 ou plus est requis : `python3 --version` doit répondre (celui fourni par macOS convient).

### Créer une vraie application `.app`

```bash
./build_mac.sh
```
Cela produit `dist/Numbered - Batch Cooking.app`, à glisser dans le dossier Applications.
Au premier lancement, macOS peut bloquer l'app (non signée) : clic droit sur l'app, puis « Ouvrir ».

## Fonctionnalités

- **Sélection** : fichiers multiples, dossier entier (avec ou sans sous-dossiers), ou glisser-déposer.
  Formats lus : JPEG, PNG, WEBP, TIFF, BMP, GIF, et HEIC (photos iPhone).
- **Recadrage** : centré selon un ratio (1:1, 4:3, 3:2, 16:9, 4:5, 2:3, 9:16), ou zone dessinée
  à la souris sur l'aperçu, appliquée à toutes les images (en proportions).
- **Remplacer le fond (détourage)** : le sujet principal (produit, personne…) est détouré
  automatiquement, en local, puis posé sur un nouveau fond : transparent, couleur unie, image de
  votre choix, ou original flouté. Option « Recadrer sur le sujet » avec une marge, idéale avec
  « Changer le ratio » pour des visuels produit carrés et centrés. Deux modèles : Rapide (ISNet)
  et Précis (BiRefNet, plus lent). Le modèle est téléchargé une seule fois (170 à 220 Mo) dans
  `~/Library/Application Support/ImageBatch/models`.
- **Nouveau fond créé par l'IA** : case « Créer le nouveau fond avec l'IA (OpenAI) » et une
  description (ex. « table en marbre blanc, lumière douce »). Le produit détouré est conservé à
  l'identique. Cumulable avec le recadrage sur le sujet et l'agrandissement par IA.
- **Changer le ratio en ajoutant du fond** : l'image entière est conservée et la toile est agrandie
  au ratio voulu, sans jamais étirer le produit. Fond au choix : prolongement du bord de l'image,
  couleur du fond détectée, couleur unie, image floutée, ou transparent.
  Deux modes IA pour un agrandissement plus propre qu'en étirant :
  « IA locale » (LaMa, gratuite et hors ligne, 92 Mo téléchargés une fois, idéale pour les fonds
  simples) et « IA OpenAI » (meilleure sur les décors complexes ; nécessite une clé API créée sur
  platform.openai.com, payante à l'usage, et les images sont envoyées à OpenAI). Dans les deux cas,
  les pixels d'origine sont conservés à l'identique : seule la zone ajoutée est générée.
  Par défaut, le fond ajouté est **généré** (« Fond généré propre ») : la couleur du bord est
  lissée puis prolongée avec un grain neutre, sans recopier les pixels, donc sans stries ni
  défauts de compression. L'IA locale l'utilise aussi automatiquement sur les fonds unis ;
  sur un décor (carrelage, bois, extérieur), elle débruite l'image avant d'agrandir.
  En position « Auto », l'app détecte par la couleur des bords si le produit touche le haut ou
  le bas (ou un côté) de l'image, et ajoute le fond uniquement du côté opposé.
  Si la détection se trompe sur une image, choisissez « Pour cette image » (haut, bas, centré…) :
  ce réglage ne concerne que l'image sélectionnée (ou les images sélectionnées), marquée ✎ dans
  la liste, et prend le pas sur le réglage du lot à l'export.
- **Retouche IA d'une zone** : ouvrez le volet, cliquez « Sélectionner une zone sur l'aperçu »
  et tracez un rectangle sur la trace à gommer. L'IA (locale, ou OpenAI avec votre clé) la
  regénère pour cette image seulement (marquée ✎), à l'aperçu et à l'export. « Annuler la
  dernière zone » et « Tout effacer » pour revenir en arrière.
- **Redimensionnement** : largeur fixe, hauteur fixe, tenir dans un cadre, dimensions exactes,
  pourcentage, avec option « ne jamais agrandir ».
- **Retouches** : rotation, miroirs, luminosité, contraste, saturation, netteté, noir et blanc.
- **Export** : JPEG, PNG, WEBP, TIFF, BMP ou format d'origine, qualité réglable, suffixe de nom,
  conservation optionnelle de l'EXIF et du profil couleur. Les originaux ne sont jamais écrasés.

Ordre d'application : recadrage → rotation/miroir → retouches → remplacement du fond → ajout de fond (ratio) → redimensionnement.

## Développement

```bash
pip install -r requirements.txt pytest
python -m pytest tests
python main.py
```

- `imagebatch/processing.py` : tout le traitement (sans interface, testable).
- `imagebatch/outpaint.py` : l'agrandissement par IA (LaMa locale ou OpenAI).
- `imagebatch/background.py` : le détourage (modèles ONNX via onnxruntime).
- `imagebatch/app.py` : l'interface.
