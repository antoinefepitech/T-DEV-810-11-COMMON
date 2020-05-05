# IA & BIG DATA
## T-DEV-810 - Groupe 11
## Antoine Falais, Léo Hamon, David Vera, Mathieu Dufour et Said Ali Hamadou

Vous trouverez le dataset original à cette adresse : [https://epitechfr.sharepoint.com/sites/TDEV810/Documents%20partages/Forms/AllItems.aspx?viewid=421729e2%2D09be%2D4217%2D9fdb%2Dc0588b1b0933&id=%2Fsites%2FTDEV810%2FDocuments%20partages%2Fdatasets%2Fchest%5FXray](https://epitechfr.sharepoint.com/sites/TDEV810/Documents%20partages/Forms/AllItems.aspx?viewid=421729e2%2D09be%2D4217%2D9fdb%2Dc0588b1b0933&id=%2Fsites%2FTDEV810%2FDocuments%20partages%2Fdatasets%2Fchest%5FXray)

Vous trouverez les datasets équilibrés à cette adresse : [https://drive.google.com/drive/folders/1p0n10Z0DnoYeLq85-hs3-eMhx_QPa9Us?usp=sharing](https://drive.google.com/drive/folders/1p0n10Z0DnoYeLq85-hs3-eMhx_QPa9Us?usp=sharing)


## Comment lancer le projet ?

Télécharger le dossier **dataset** du google drive (comprenant dataset_2_classes + dataset_3_classes) et le mettre à la racine du projet. Le dataset_2_classes correspond à un dataset équilibré NORMAL/PNEUMONIA, le dataset_3_classes correspond à un dataset équilibré NORMAL/VIRUS/BACTERIA.

---

#### NORMAL/PNEUMONIA
Le notebook "notebook_2_classes.ipynb" permet de lancer un modèle _VGG16_ pour déterminer des poumons NORMAUX ou avec PNEUMONIE (utilise le dataset_2_classes).

---

#### NORMAL/BACTERIA/VIRUS
Le "notebook_3_classes.ipynb" permet le lancer un modèle pour déterminer des poumons NORMAUX, avec BACTERIE ou avec VIRUS (utilise le dataset_3_classes).

**ATTENTION : Ce notebook regroupe 3 modèles.**
Au moment du lancement, éxecuter toutes les cellules dans l'ordre, sauf la 4, 5, 6 et la dernière.
La cellule 4 correspond à un _VGG16_, la cellule 5 correspond à un _ResNet18_ et la cellule 6 correspond à un _ResNet34_.
Choisir un modèle en éxécutant sa cellule correspondante. Puis lancer la toute dernière cellule du notebook.

cellule 4 -> get_model_vgg_16(...)

cellule 5 -> get_model_resnet_18(...)

cellule 6 -> get_model_resnet_34(...)

Voici l'ordre d'éxécution : 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 4/5/6 (au choix), 14.