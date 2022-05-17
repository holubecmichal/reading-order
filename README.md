# Uspořádání fragmentů textu s pomocí jazykového modelu

Tento repozitář je součástí diplomové práce na téma
"Uspořádání fragmentů textu s pomocí jazykového modelu".
Cílem diplomové práce bylo vytvořit jazykový model a experimentálně 
ověřit, zda je možné jazykový model použít při definici posloupnosti čtení
(Reading Order).
K tomuto účelu byly implementovány tři metody: jazyková analýza, prostorová analýza
a kombinovaná analýza.

Program pracuje s digitalizovanými novinovými stránkami, jejichž struktura a OCR výstup
je uložen v PageXML souboru.
PageXML soubor obsahuje OCR výstup, může obsahovat také souřadnice polygonu identifikovaných
objektů, jako jsou odstavce, řádky, slova.
Dále lze do tohot souboru uložit strukturu ReadingOrder, která je v tomto projektu
použita jako ground truth.

Součástí respozitáře jsou skripty pro trénování modelu, ověření jeho chování, vizualizaci
naměřených dat a podobně.
- `language_model_train.py`
- `language_model_evaluate.py`
- `language_model_graphs.py`

Dále jsou součástí metriky, kterými se měří úspěšnost identifikované posloupnosti
proti ground truth.

- `reading_order/metric/prima.py`
- `reading_order/metric/recall.py`

V neposlední řadě jsou součástí skripty pro vyhodnocení experimentů, jejichž cílem
bylo ověřit úspešnost implementovaných metod pro identifikaci posloupnosti čtení.

- `reading_order_eval.py`
- `reading_order_graphs.py`

Z licenčních důvodů nejsou v tomto repozitáři zahrnuty
xml soubory, nad kterými byly experimenty provedeny.
Součástí také není trénovací korpus.
Tyto soubory jsou však součástí diplomové práce a jsou
umístěny na příslušném médiu.

### Jazyková analýza
Cílem jazykové analýzy je identifikovat posloupnost čtení na základě textového obsahu 
regionů. Nejčastěji se jedná o výstup OCR. Pracuje s jazykovým modelem postaveným
na LSTM.
Principem je odhad podmíněných pravděpodobností jednotlivých regionů.
Ty regiony, kterým jazykový model přiřadí nejvyšší pravděpodobnost, jsou spojeny do posloupnosti.
Takto jsou zpracovány všechny regiony, dokud není definovaná celková posloupnost čtení.

Jazykovou analýzu implementuje třída `LmAnalyzer`. Analýzu lze spustit voláním metody
`analyze(doc: Document)`, kde vstupním parametrem je analyzovaný dokument.
Chování lze ovlinit metodami `set_hard_limit(tokens: int)` a `set_score_hard_limit(tokens: int)`.
První metoda používá geometrický průměr vypočtených pravděpodobností a používá 
počet tokenů, který je definován parametrem při volání metody.
Druhá metoda počítá skóre, které lze také interpretovat pro definici posloupnosti.
Vhodný počet tokenů byl experimentálně ověřen a pro jednotlivé metody jsou následující:

- `set_hard_limit`: 2 až 3 tokeny
- `set_score_hard_limit`: 4 až 6 tokenů

### Prostorová analýza
Prostorová analýza pracuje s prostorovými informacemi jednotlivých regionů,
pro které definuje posloupnost čtení. Práce implementuje více přístupů.

První je diagonální analýza, která pracuje se vztahy, které jsou identifikovány instancí
třídy `DocTBRR`. Ta identifikuje celkem 13 vztahů jak pro osu x a osu y.
Dle pravidel pro jednotlivé vztahy pak diagonální analýza `DiagonalAnalyzer` identifikuje 
posloupnost čtení.

Druhým přístupem je sloupcová analýza `ColumnarAnalyzer`. 
Pracuje obdobně jako analýza diagonální s tím
rozdílem, že před analýzou posloupnosti jsou pro jednotlivé regiony identifikovány
sloupce, ve kterých se regiony pravděpodobně vyskytují. Množina regionů v jednom sloupci
je poté v analýze nahrazena tímto sloupcem. To snižuje počet regionů, které
je nutné analyzovat a vylepšuje výsledek. Po identifikaci posloupnosti 
jsou sloupce nahrazeny opět jednotlivými regiony, mezi kterými je implicitně uvažovaná
posloupnost odshora sloupce dolů.

### Kombinovaná analýza
Kombinovaná analýza `ColumnarLmAnalyzer` kombinuje prostorovou sloupcovou analýzu 
s jazykovou analýzou. Nejprve je provedena prostorová analýza, jejíž výsledek je
v zápětí korigován jazykovým modelem.

### Výsledky
Výstup experimentů je přiložen v samostatném adresáři `Experiments` .
Nejnižších výsledků dosáhla jazyková analýza, která dosáhla maxima 57,6%.
Následuje prostorová analýza s maximem 91,6%. Kombinovaná analýza dosáhla 92,9%.
Závěrem lze říct, že samostatné použití jazykového modelu pro identifikaci posloupnosti
čtení, není vhodné, pokud neni doplněno o další informace, jako je to právě v kombinované
analýze.

## Trénování jazykového modelu
Jazykový model lze trénovat pomocí skriptu `language_model_train.py`.
Model, použitý v této práci, byl trénovaný na výňatku české Wikipedie.
Dosáhl perplexity 31,79 a proběhlo 28 epoch. Model byl trénován s následujícími
parametry:

```bash
python language_model_train.py --emsize=400 --nhid=1700 --cuda=1 --vocab_size=20000
```

Trénovací skript byl převzat z https://github.com/pytorch/examples/tree/main/word_language_model.

### Vyhodnocení jazykového modelu
Vyhodnocení jazykového modelu implementuje skript `language_model_evaluate.py`.
Ten provede ověření chování modelu při při různých konfiguracích délky textu,
respektive tokenu.
Pro vyhodnocení je použitý rozsah 1-64 tokenů jak pro incializaci modelu, tak pro vyhodnoceni.
Výsledky jsou uloženy do pkl souboru, které jsou zpracovány samostatným skriptem.

```bash
python language_model_evaluate.py
```

### Vizualizace naměřených dat
Naměřená data lze vyzualizovat pomocí heatmapy. O to se stará skript `language_model_graphs.py`.
Skript používá uložená data ve formátu pkl.

```bash
python language_model_graphs.py
```

## Experimenty
Soubory s experimenty jsou uloženy ve složce `experiments` a jsou rozděleny na jednotlivé
datasety.
Pro spuštení experimentu a vyhodnocení všech tří představených metod je nutné zavolat
skript `reading_order_eval.py` s parametrem s cestou ke složce s dokumenty k vyhodnocení.
Výsledky jsou uloženy ke každé analyzované stránce ve podobě pkl.

```bash
python reading_order_eval.py --path=./experiments/hn
```

Po zpracování a uložení dat je možné výsledky vizualizovat pomocí skriptu `reading_order_eval_graphs.py`.
Skript očekává parametrem cestu k složce s uloženými výsledky.
Výstupem skriptu jsou pdf soubory s grafy.
Graf vizualizuje úspěšnost jednotlivých metod při identifikaci posloupnosti čtení.

```bash
python reading_order_eval_graphs.py --path=./experiments/hn
```

## Process
I přes to, že je práce převážně experimentální, vytvořil jsem, pro 
vyzkoušení jednotlivých metod, samostatný skript `process.py`.
Ten přijímá dva argumenty: cestu k souboru, který bude analyzován a metodu, kterou bude analyzován.
Jednotlivé metody, které jsou k dispozici:

```
LH - language analyse with language model, hard token decision
LS - language analyse with language model, score
SD - spatial diagonal analyse
SC - spatial columnar analyse
CH - combined analyse, hard token decision
CS - combined analyse, score
TB - top-to-bottom analyse
```

Lze nastavit počet tokenů, které se použijí k analýze stejně jako ground truth.
Pokud je nastaven ground truth, bude identifikovaná posloupnost porovnána a změřena
právě proti ground truth.

Výsledkem skriptu je vizualizace posloupnosti, případně vyhodnocení posloupnosti vůči 
ground truth.

```bash
python process.py ./experiments/hn/hn-12-1-2022-04.xml CS --tokens=5 --g=./experiments/hn/hn-12-1-2022-04.xml
```

Více možností nastavení viz. 

```bash
python process.py --help
```
