# Reactie op de review van optimize.py, vlinder_demo.ipynb en trbs.py

Per punt uit `review_feedback_vlinder_optimizer.docx` (Louise) staat hieronder of en hoe het is
verwerkt. De feedback van Ruben (`joep_feedback.docx`) staat in deel 5, omdat die deels over
dezelfde regels gaat.

**Uitgangspunt bij de refactor:** het rekengedrag mocht niet veranderen, behalve waar de
feedback daar expliciet om vraagt. Dat is nagemeten. Twaalf combinaties van case en methode
(beerwiser, refugee, IZZ x slsqp, basin-hopping, GA, MDBH) leveren na de herstructurering exact
dezelfde appreciatie en exact hetzelfde aantal objective-evaluaties als ervoor. De enige
bedoelde uitzondering is de budget-bepaling, zie punt 1.4.

**Status van de suites:** 255 tests slagen. Eén test faalt, `test_make_report.py::test_create_report[Optimistic]`,
op een bestaande Windows-bug: `strftime("%H:%M:%S")` zet dubbele punten in een bestandsnaam, wat
Windows niet toestaat. Die stond er al en staat los van deze wijzigingen.

---

## Deel 1: optimize.py

### 1.1 Introductie / module docstring

**Verwerkt.** De oorspronkelijke openingszin staat terug:
"This module contains the Optimize class, which performs grid search optimization to maximize the
appreciation of decision-maker options." Daarna volgt een genummerde uitleg van de modulaire
opbouw: de losse evaluatiefuncties, de resultaatcontainer, de base class, de vijf solvers en de
orchestrator. De laatste alinea legt uit dat de continue solvers op de capped simplex werken en
grid search niet.

### 1.2 Evaluate-functies

**Verwerkt.** `evaluate_output` heet nu `evaluate_and_appreciate`. De int64-comment is ingekort
tot vrijwel jouw formulering:

> DMO values that are whole numbers get imported as int64. Writing a float allocation into an int
> row rounds it silently, which breaks gradient-based solvers. Cast to float to avoid this.

### 1.3 OptimizationResult

**Verwerkt.** Toegevoegd: `budget`, `budget_spent`, `scenario`, `timestamp` en
`calculation_time`. Dat laatste veld is de hernoeming van het bestaande `wall_time_s`, zodat er
geen twee velden voor hetzelfde bestaan; de twee experimentscripts die dat veld lazen zijn
meegenomen.

Voor de weergave is er `result.summary()`, die alle velden als leesbaar blok teruggeeft. In de
demo wordt hetzelfde als tabel getoond, zie deel 2.

### 1.4 Bestandsvolgorde

**Verwerkt**, in de volgorde die je voorstelt:

| # | Component | regel |
| --- | --- | --- |
| 1 | `evaluate_and_appreciate()` | 46 |
| 2 | `evaluate_allocation()` | 85 |
| 3 | `OptimizationResult` | 106 |
| 4 | `BaseSolver` | 151 |
| 5 | `GridSearch(BaseSolver)` | 270 |
| 6 | `SLSQPSolver(BaseSolver)` | 427 |
| 7 | `BasinHoppingSolver(SLSQPSolver)` met `_RandomFeasibleHop` als geneste class | 541 |
| 8 | `GeneticAlgorithmSolver(BaseSolver)` | 669 |
| 9 | `MdbhSolver(BaseSolver)` | 790 |
| 10 | `Optimize` (orchestrator) | 931 |

`MdbhSolver` staat niet in jouw indeling. Dat is de methode uit mijn thesis. Hij staat nu als
gewone solver in de rij, zodat de thesis-experimenten op dezelfde infrastructuur blijven draaien.
Als jullie hem liever uit het pakket houden, haal ik hem eruit; dat is één class verplaatsen plus
het registry-item.

### 1.5 Class Optimize (orchestrator)

**Verwerkt.** `Optimize` bevat alleen nog het kiezen en aanroepen van een solver plus het
vergelijken van meerdere methodes. Grid search is een eigen class geworden. Alle vier de plekken
die `best_x` op hun eigen manier terugschreven zijn vervangen door één
`BaseSolver._write_back_result(dmo_name, allocation)`.

**`_infer_budget` aangepast, met gevolgen.** Het budget is nu het maximum van de rijsommen over
alle DMO's, in plaats van de rijsom van de eerste DMO. Dit is de enige wijziging die uitkomsten
verandert, en die verandering is groot:

| case | budget oud | budget nieuw | appreciatie oud | appreciatie nieuw |
| --- | --- | --- | --- | --- |
| NEMO | 0,00 | 3.407,00 | `nan` | 69,35 |
| DSM | 0,50 | 1,50 | 88,03 | 95,73 |
| beerwiser, refugee, IZZ | ongewijzigd | ongewijzigd | ongewijzigd | ongewijzigd |

Bij NEMO is de eerste DMO de "niets doen"-optie, dus het budget was nul en elke continue methode
gaf `nan`. Bij DSM mocht de optimizer een derde van het echte budget uitgeven. Het zijn dus geen
cosmetische wijzigingen maar echte fouten die eruit zijn.

Ter controle: alle synthetische cases houden hetzelfde budget, dus de lopende thesis-studie is
niet geraakt.

### 1.6 Grid Search

Alle punten verwerkt:

- Comment ingekort tot "Evaluate and appreciate without changing self.input_dict during the loop.
  Only the best result is written back after the loop finishes."
- Grid roept nu `_prepare_input_dict` aan in plaats van zijn eigen kopie te maken.
- `len(self.input_dict["internal_variable_inputs"])` vervangen door `self._k`.
- De printstatements en `@suppress_print` zijn weg.
- Grid loopt via `run()` en geeft een `OptimizationResult` terug; `optimize_single_scenario` is
  verdwenen.
- De class-docstring legt uit dat grid op de gewone simplex `sum(x) = B` werkt en dus een optimum
  dat budget laat liggen niet kan vinden.
- De guard die een bestaande DMO-naam weigerde is weg. Er is een test bij die twee keer achter
  elkaar onder dezelfde naam optimaliseert.

### 1.7 SLSQP

**Verwerkt.** `scipy~=1.13` staat nu bij de dependencies in `pyproject.toml`. De verwijzingen
naar "W2 thesis work" en "W3+" zijn verdwenen; in `optimize.py` en `trbs.py` staat geen enkele
verwijzing meer naar interne experimenten.

### 1.8 Basin-Hopping

Alle punten verwerkt:

- `BasinHoppingSolver` is een eigen class met een eigen docstring, dus het herkenningspunt is er.
- De docstring is herschreven, de verwijzing naar exp01 is weg.
- `_CappedSimplexStep` heet nu `_RandomFeasibleHop` en staat als geneste class in
  `BasinHoppingSolver`.
- De docstring zegt expliciet dat dit eigenlijk SLSQP-hopping is: "Because the local solve is
  SLSQP, this is really SLSQP-hopping: same model, same feasible set, wrapped in a search for
  other optima." De naam is ongewijzigd.
- De dubbele definitie van constraints, bounds en options is samengetrokken in
  `SLSQPSolver._slsqp_minimizer_kwargs()`, die zowel de losse solve als elke hop gebruikt.
- De deepcopy per hop is opgelost, zie 1.9.

### 1.9 Performance: de deepcopy (ook Rubens punt)

**Verwerkt, en het bleek groter dan alleen de deepcopy.** Een profiel van één
objective-evaluatie liet drie kosten zien die per evaluatie werden betaald terwijl ze per run
constant zijn:

| | beerwiser | IZZ |
| --- | --- | --- |
| grenzen opnieuw opbouwen in `Appreciate.__init__` (via een pandas DataFrame) | 48% | 32% |
| alle DMO's evalueren terwijl er één nodig is | 23% | 45% |
| `deepcopy` van de hele case | 20% | 17% |

Alle drie zijn aangepakt: een ondiepe kopie plus een kopie van alleen
`decision_makers_option_value`, één DMO evalueren in plaats van alle, en de grenzen één keer per
run berekenen in plaats van per evaluatie.

Resultaat per evaluatie: **beerwiser 2,838 ms → 0,213 ms (13,3x), IZZ 5,380 ms → 1,052 ms (5,1x)**.
De uitkomsten zijn bit-identiek: over vijf cases, alle scenario's en 120 allocaties per scenario
is het grootste verschil in appreciatie exact 0.

### 1.10 Genetic Algorithm

**Verwerkt.** Eigen class met eigen docstring, `_sbx_pair` en `_polynomial_mutation` staan
eronder in dezelfde class, en de uitleg is korter en zonder jargon herschreven.

---

## Deel 2: vlinder_demo.ipynb

**Let op, mogelijk misverstand.** Jouw punten verwijzen naar "sectie 6.3". Die nummering hoort bij
de 58-cellen demo (`20260217_vlinder_demo.ipynb`), die niet in de repository staat. De demo die
wel in git zit en die met het pakket wordt meegeleverd is een oudere versie van 36 cellen. Ik heb
de versie in de repository bijgewerkt, omdat dat de versie is die gebruikers krijgen. **Vraag:
welke van de twee is de canonieke demo?** Als het de 58-cellen versie is, verhuis ik die eerst
naar de repository en pas ik daar dezelfde wijzigingen toe.

In de bijgewerkte demo:

- **Uitleg over de methodes:** de sectie begint nu met een tabel van de beschikbare methodes, wat
  ze doen en wanneer ze passen, plus de uitleg dat `auto` zelf kiest en dat je een lijst kunt
  meegeven om er meerdere te draaien.
- **Σx ≤ B geldt niet voor grid:** er staat een aparte alinea die uitlegt dat de continue methodes
  hoogstens het budget mogen uitgeven en grid precies het budget, en dat een optimum dat geld laat
  liggen daardoor buiten het zoekgebied van grid valt.
- **De auto-print is leesbaar gemaakt.** In plaats van
  `[auto] selected basin_hopping: key outputs are not affine...` komt er nu:

  ```
  Automatic method selection chose basin-hopping.
    The case has 2 levers and a budget of 300,000.00.
    Its appreciation surface has possibly more than one optimum.
    Diagnosed in 36 evaluations.
    Why: key outputs are not affine in the allocation (residual 4.9e-02); floor clipping on 91%
    of probe points, so the surface can be multimodal and a local solver can settle in the wrong basin.
  ```

- **Het resultaat als tabel**, inclusief scenario, timestamp en budget:

  | | value |
  | --- | --- |
  | method | basin_hopping |
  | scenario | Base case |
  | optimized DMO | Optimized (basin-hopping) (Base case) |
  | appreciation | 65.7116 |
  | budget spent | 300,000.00 of 300,000.00 |
  | allocation | [25000.0, 275000.0] |
  | evaluations | 7680 |
  | calculation time | 1.61 s |
  | run at | 2026-08-10T14:33:34+00:00 |

- **De DMO-naam bij `method="grid"`** komt inderdaad goed mee na de modulaire opbouw. Nagemeten:
  zonder naam wordt het `Optimized (grid) (Base case)`, met `dmo_name="My Grid Run"` wordt het
  `My Grid Run (Base case)`, en in beide gevallen staat die naam op de case.
- **Het scenario zit in de DMO-naam**, voor alle methodes, ook als de gebruiker zelf een naam
  opgeeft.

### De vraag over de boundaries bij een tweede optimalisatie

**Onderzocht, en het gebeurt niet.** Ik heb beerwiser twee keer achter elkaar geoptimaliseerd, met
een evaluate en appreciate ertussen, en de grenzen vergeleken:

```
key_output_start na run 1: [3.4884, 1227272.7273, 0.037]
key_output_start na run 2: [3.4884, 1227272.7273, 0.037]
key_output_end   na run 1: [17.442, 6818181.8182, 0.0547]
key_output_end   na run 2: [17.442, 6818181.8182, 0.0547]
```

Beide runs geven dezelfde appreciatie en dezelfde allocatie. De reden is dat de eerste
optimalisatie `key_output_automatic` op nul zet en de grenzen wegschrijft; daarna leidt
`_get_start_and_end_points` ze niet meer af uit de output maar leest het die vastgezette waarden.
Dat lijkt me ook het gewenste gedrag, want alleen zo zijn twee geoptimaliseerde DMO's op dezelfde
appreciatieschaal te vergelijken.

Wel een gevolg om te weten: de grenzen worden nooit meer opgerekt. Als een geoptimaliseerde
allocatie een key output buiten het oorspronkelijke bereik oplevert, wordt die op 0 of 100
afgekapt. Bij beerwiser gebeurt dat ook echt, want het optimum ligt precies op zo'n knik.

### De optionele max_time parameter

**Niet gedaan deze ronde**, in overleg met Joep buiten scope gehouden omdat het geen kleine
toevoeging is: machinesnelheid meten, dat vertalen naar instellingen per methode, en bepalen wat
er gebeurt als meerdere solvers tegelijk draaien. Staat op de issue-lijst in deel 4.

De cijfers laten wel zien dat de behoefte er is. Op NEMO kost `method="auto"` nu 130 seconden.

---

## Deel 3: trbs.py

- **Eén naamgeving.** Alleen `dmo_name` nog. De backward-compatibility regels en
  `_resolve_grid_dmo_name()` zijn verwijderd, inclusief de bijbehorende `pandas`- en
  `CaseError`-afhankelijkheid op dat pad.
- **Scenario in de DMO-naam.** Verwerkt, in de orchestrator zodat het ook geldt als de solver zijn
  eigen standaardnaam gebruikt.
- **kwargs bij meerdere methodes.** Er zijn nu twee vormen. Een parameter die je direct meegeeft
  gaat naar elke methode die draait, wat prettig is bij één methode of bij iets dat ze delen zoals
  `seed`. Voor parameters die per methode verschillen is er `method_kwargs`:

  ```python
  case.optimize(
      "Base case",
      method=["grid", "slsqp"],
      method_kwargs={"grid": {"max_combinations": 1000}, "slsqp": {"n_starts": 50}},
  )
  ```

  Een instelling per methode wint van een gedeelde. Instellingen voor een methode die niet draait
  geven een foutmelding, zodat een typefout niet stilletjes wordt genegeerd. De
  solver-parameters zijn meteen keyword-only gemaakt, waardoor een subclass niet meer positioneel
  kan botsen met zijn parent.

  **Vraag:** dit maakt de API wel zwaarder dan de simpele variant waarin alles naar elke methode
  gaat. Wil je hem zo houden?
- **Compatibel met de front-end.** `case.optimize(...)` geeft weer de `input_dict` terug, precies
  zoals Ruben vraagt voor Papilio. Het volledige `OptimizationResult` staat na afloop op
  `case.optimization_result`. Zo hoeft er in de front-end niets te veranderen en is de rijkere
  informatie toch beschikbaar.

---

## Deel 4: algemene feedback

- **Structuur en modulariteit:** verwerkt, zie 1.4.
- **Pleisters van Claude:** de twee die je noemt zijn weg (`new_dmo_name`-fallback en
  `_resolve_grid_dmo_name`). Ik ben de rest ook langsgelopen; de enige constructie die er nog op
  lijkt is dat elke solver `**_ignored` in zijn signatuur heeft, wat nodig is omdat één lijst
  kwargs langs meerdere methodes gaat. Dat is nu bewust en gedocumenteerd, geen workaround.
- **Taalgebruik in comments:** alle verwijzingen naar exp01, W2 en W3+ zijn weg uit `optimize.py`
  en `trbs.py`.
- **Testen via de front-end:** gedaan. De hele demo is headless uitgevoerd. De optimalisatieketen
  loopt schoon: het optimum `[25000, 275000]` met appreciatie 65,711590, en na `evaluate()` en
  `appreciate()` komt de case op dezelfde waarde uit (verschil 1,3e-11). De geoptimaliseerde DMO
  wordt daarna terecht de hoogst gewaardeerde optie.

  **Twee bestaande bugs gevonden** in dezelfde run, allebei los van deze wijzigingen en in
  bestanden die ik niet heb aangeraakt:
  1. `case.visualize('dependency_graph', ...)` faalt op Windows met een `UnicodeEncodeError`.
     `pyvis` schrijft de HTML zonder expliciete encoding weg, waardoor Windows er cp1252 van
     maakt (`visualize.py`, regel 683).
  2. `case.make_report(...)` faalt op Windows, de bekende dubbele punten uit `strftime` in de
     bestandsnaam (`make_report.py`).

  Zal ik daar aparte issues van maken, of los ik ze meteen op?
- **test_optimize.py:** opgesplitst langs de structuur van de module, wat volgens mij het
  antwoord is op "hoe houd je 500+ regels behapbaar":

  | bestand | regels | inhoud |
  | --- | --- | --- |
  | `tests/conftest.py` | 89 | gedeelde fixtures |
  | `tests/test_optimize_evaluation.py` | 121 | de losse evaluatiefuncties en de resultaatcontainer |
  | `tests/test_optimize_solvers.py` | 362 | één sectie per solver, plus een gedeeld contract dat elke solver via `parametrize` doorloopt |
  | `tests/test_optimize_orchestrator.py` | 182 | methodekeuze, dispatch en de keten via `case.optimize()` |

  Het gedeelde contract onderaan `test_optimize_solvers.py` scheelt het meeste: drie tests die
  over alle vijf solvers heen lopen en controleren dat het resultaat de gedeelde velden vult, dat
  het antwoord op de case belandt en dat een run met dezelfde seed reproduceerbaar is.

  Daarbij kwam een probleem boven water: de dictionaries in `tests/params.py` zijn gedeeld en
  worden door testbestanden aangepast. De oude optimizer-tests slaagden alleen doordat
  `test_appreciate.py` eerst had gedraaid en `highest_weighted_dmo` had ingevuld. De fixtures
  bouwen die afgeleide velden nu zelf op, zodat elk bestand ook los draait. Dat is nagemeten.

### Openstaande punten voor issues

Deze heb ik bewust niet in deze ronde verwerkt:

1. `max_time` met automatische vertaling naar instellingen per methode.
2. Detectie of een case een budget-allocatiecase is; dit wordt nu nergens gevalideerd.
3. NEMO heeft binaire levers. Met de budget-fix geeft de optimizer nu wel een antwoord (69,35),
   maar er is nog geen validatie op binaire of discrete inputs, dus dat antwoord is niet
   noodzakelijk zinvol. Dit lijkt me het punt met het meeste risico van de vijf.
4. Multi-scenario optimalisatie met een gewogen doelfunctie.
5. Andere optimizers.

Zeg maar of je wilt dat ik deze als GitHub-issues aanmaak.

### IZZ geeft het volledige budget uit

**Klopt, en nagemeten.** Zowel SLSQP als basin-hopping komen op precies 100,00 van 100,00, dus
100% van het budget. De uitspraak dat een optimum budget kan laten liggen staat nu algemeen
geformuleerd in de docstring van `SLSQPSolver`, zonder IZZ als voorbeeld te noemen.

---

## Deel 5: de feedback van Ruben

**Verwerkt:**

- **De deepcopy** (zijn punt over regel 56). Zie 1.9. Ook zijn observatie dat het alleen om
  `decision_makers_option_value` en de index gaat klopte precies.
- **Het return contract.** `optimize()` geeft weer de `input_dict` terug. Zie deel 3.
- **"model_selection impliceert dat basin hopping en slsqp verschillen, maar het onderliggende
  model is hetzelfde."** Dat staat nu expliciet in de docstring van `BasinHoppingSolver`.

**Onderzocht en gemeten, nog niet ingebouwd:**

- **Early stopping via `niter_success`.** Gemeten over twaalf cases: met `niter_success=5` gaan er
  46 tot 68% van de evaluaties af, en op tien van de twaalf cases blijft het antwoord exact
  gelijk. Op refugee kost het 0,041 appreciatiepunt.

  Wel een waarschuwing: een consensusregel *tussen* chains is gevaarlijk. Op refugee zijn twee
  chains het na 84,50 met elkaar eens en stopt zo'n regel daar, terwijl het optimum 96,87 is. Dat
  is precies de ene case waar je basin-hopping voor nodig hebt.

- **Zijn idee voor concave gevallen** ("run slsqp n keer, aggreeable optima, kies de hoogste") is
  gemeten en werkt, mits n groot genoeg is. Met acht starts markeert de spreiding tussen de
  gevonden waarden precies de twee cases die een globale methode nodig hebben (refugee 8,76 en
  DSM 0,42) en laat het de andere tien door (spreiding hoogstens 0,0102). Met vier starts mist hij
  refugee, want die vier landen dan allemaal in dezelfde val.

  Dit is volgens mij de grootste winst die er nog ligt. `auto` kiest nu op alle vijf de cases
  basin-hopping en is daarmee steeds de duurste optie, zonder noemenswaardig beter te zijn dan
  gewone SLSQP (zie de tabel hieronder).

- **Parameters afhankelijk van k maken.** Gemeten, en k blijkt de verkeerde voorspeller. Het
  aantal starts dat nodig is om binnen 0,1 van het optimum te komen:

  | case | k | starts nodig |
  | --- | --- | --- |
  | Synthetic convex k9, convex curved k9 | 9 | 1 |
  | Synthetic k3-varianten | 3 | 1 |
  | beerwiser | 2 | 1 |
  | IZZ | 9 | 1 |
  | DSM | 7 | 14 |
  | refugee | 5 | niet binnen 30 |

  Zijn conclusie over beerwiser klopt dus, één start is genoeg, maar niet omdat k=2 is: de
  convexe cases met k=9 hebben ook aan één start genoeg, en refugee met k=5 komt er in dertig
  starts niet. Het regime bepaalt het, niet de dimensie. Een regel `n_starts = f(k)` zou beerwiser
  goed bedienen en refugee juist verhongeren.

  De probe zelf schaalt overigens al lineair met k, ongeveer 9k+9 evaluaties: 36 bij k=2 en 99 bij
  k=9. Dat is verwaarloosbaar en hoeft niet aangepast.

**Niet gedaan, wel genoteerd:** warm start van SLSQP in plaats van nieuwe basin-hopping-iteraties.
Adaptieve stapgrootte gebeurt al: scipy stelt het `stepsize`-attribuut van de hop zelf bij naar
een doelacceptatieratio.

---

## Vergelijkingstabel

Je vroeg om een tabel met methodes, rekentijd en appreciatie per case, en om de code erachter.
Die code staat nu in de repository als `experiments/method_comparison.py` en is opnieuw te
draaien met:

```
python experiments/method_comparison.py
python experiments/method_comparison.py --cases beerwiser refugee --seed 7
```

Elke methode draait op zijn eigen standaardinstellingen, dus dit is wat een gebruiker krijgt.

| case | k | methode | appreciatie | evaluaties | seconden |
| --- | --- | --- | --- | --- | --- |
| beerwiser | 2 | grid | 65,711590 | 3.001 | 0,97 |
| | | slsqp | 65,711590 | 2.251 | 0,37 |
| | | basin_hopping | 65,711590 | 3.233 | 0,48 |
| | | genetic_algorithm | 65,701420 | 3.050 | 0,31 |
| | | mdbh | 65,711519 | 6.124 | 0,53 |
| | | auto (koos basin_hopping) | 65,711593 | 7.991 | 1,67 |
| refugee | 5 | slsqp | 96,824258 | 14.718 | 6,64 |
| | | basin_hopping | 96,870901 | 12.968 | 6,47 |
| | | genetic_algorithm | 96,867831 | 3.050 | 1,05 |
| | | mdbh | 96,818802 | 10.267 | 3,65 |
| | | auto (koos basin_hopping) | 96,857100 | 32.737 | 14,04 |
| IZZ | 9 | slsqp | 69,190547 | 30.764 | 14,85 |
| | | basin_hopping | 69,190473 | 29.290 | 12,81 |
| | | genetic_algorithm | 69,172811 | 3.050 | 1,29 |
| | | mdbh | 69,190002 | 14.885 | 6,26 |
| | | auto (koos basin_hopping) | 69,190537 | 77.276 | 36,43 |
| DSM | 7 | slsqp | 95,726092 | 6.331 | 1,96 |
| | | basin_hopping | 95,726092 | 5.634 | 2,33 |
| | | genetic_algorithm | 95,707623 | 3.050 | 1,26 |
| | | mdbh | 95,726086 | 16.687 | 4,65 |
| | | auto (koos basin_hopping) | 95,726092 | 15.515 | 6,27 |
| NEMO | 18 | slsqp | 69,348949 | 126.666 | 52,83 |
| | | basin_hopping | 69,348955 | 139.106 | 59,87 |
| | | genetic_algorithm | 68,642884 | 3.050 | 1,31 |
| | | mdbh | 69,348691 | 52.020 | 18,35 |
| | | auto (koos basin_hopping) | 69,348942 | 343.641 | 130,60 |

Grid search draait alleen op beerwiser. Boven drie levers wordt hij onbruikbaar, omdat elke
combinatie wordt uitgeklapt naar al zijn permutaties: de kosten groeien met k faculteit. Bij negen
levers is dat 362.880 permutaties per combinatie. Dat is een eigenschap van de baseline, en
precies de reden dat de continue methodes er zijn.

Wat er verder uit de tabel valt af te lezen:

1. Op vier van de vijf cases komen alle continue methodes binnen 0,05 van elkaar. Alleen refugee
   scheidt de methodes echt, en daar is basin-hopping inderdaad de beste.
2. `auto` kiest op alle vijf de cases basin-hopping en is daardoor steeds de duurste optie, tot
   343.641 evaluaties op NEMO, zonder beter te zijn dan gewone SLSQP. Dat is hetzelfde punt dat
   Ruben maakt, nu op de echte cases.
3. Het genetische algoritme is consequent het goedkoopst en op refugee verrassend goed, maar
   verliest overal een klein beetje nauwkeurigheid.
