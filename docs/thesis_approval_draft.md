# Thesis Topic Approval Request — Concept

## Titel (voorlopig)
Optimale Kapitaalallocatie in een Multi-Criteria Beslissingsondersteunend Simulatiemodel voor Verantwoord Ondernemen

---

## Probleembeschrijving (~500 woorden)

Organisaties staan steeds vaker voor investeringsbeslissingen die niet alleen op financieel rendement beoordeeld worden, maar ook op sociale en milieu-impact. De opkomst van ESG-criteria (Environmental, Social, Governance) en toenemende regelgeving rondom duurzaamheidsrapportage (CSRD, ISSB) dwingen besluitvormers om meerdere, vaak conflicterende doelstellingen tegelijkertijd te wegen. Dit creëert een inherent multi-criteria beslissingsprobleem: hoe verdeel je een beperkt investeringsbudget over meerdere opties wanneer de uitkomsten op financiële, sociale en ecologische KPI's onderling afhankelijk zijn en mogelijk tegenstrijdig?

De Responsible Business Simulator (tRBS), ontwikkeld door Vlinder en gebruikt in de adviespraktijk van PwC, biedt hiervoor een simulatieraamwerk. Het model evalueert investeringsopties over meerdere scenario's en KPI's, waarbij waarderingsfuncties (lineair en sinusoïdaal) ruwe KPI-waarden omzetten naar genormaliseerde appreciatiescores op een schaal van 0 tot 100. Een gewogen aggregatie over KPI's en scenario's levert een totaalscore per investeringsoptie op.

Het huidige model kent echter een fundamentele beperking: het vergelijkt uitsluitend een eindig aantal vooraf gedefinieerde investeringsallocaties. De besluitvormer specificeert discrete opties (bijvoorbeeld 100% in A, 50/50, of 100% in B) en het model berekent welke optie de hoogste totale appreciatie oplevert. De bestaande optimalisatiefunctionaliteit gebruikt een brute-force grid search, gebaseerd op combinatorische enumeratie (stars-and-bars), waarbij alle mogelijke discrete verdelingen tot een bepaalde stapgrootte worden doorgerekend. Deze aanpak lijdt onder de vloek der dimensionaliteit: bij meer dan twee investeringsopties of fijnere resolutie explodeert het aantal evaluaties exponentieel (Bergstra & Bengio, 2012), terwijl grove discretisatie potentieel betere allocaties mist.

Dit onderzoek adresseert deze beperking door de discrete vergelijking te vervangen door continue optimalisatie van de kapitaalallocatie. Concreet wordt het allocatieprobleem geformuleerd als een niet-lineair programmeringsprobleem (NLP), waarbij de verdeling van budget over investeringsopties wordt geoptimaliseerd onder een simplexrestrictie (alle allocaties sommeren tot het totaalbudget, niet-negativiteit). Een belangrijk methodologisch vraagstuk is de niet-convexiteit van de doelfunctie: de sinusoïdale waarderingsfuncties in tRBS creëren een multimodaal optimalisatielandschap, wat betekent dat lokale optimalisatiemethoden niet garanderen dat het globale optimum wordt gevonden (Boyd & Vandenberghe, 2004). Dit vereist een zorgvuldige vergelijking van oplossingsmethoden.

De relevantie van dit onderzoek is tweeledig. Wetenschappelijk slaat het een brug tussen Multi-Criteria Decision Analysis (MCDA), die traditioneel uitgaat van een eindige verzameling alternatieven (Greco et al., 2016), en continue portfolio-optimalisatie, zoals het ESG-efficiënte-frontierconcept van Pedersen et al. (2021). Praktisch biedt het PwC en andere adviesorganisaties een methodologisch onderbouwd instrument waarmee klanten niet alleen de beste van een aantal vooraf bepaalde opties identificeren, maar daadwerkelijk de optimale verdeling van middelen vinden — inclusief inzicht in de gevoeligheid van deze verdeling voor veranderende omstandigheden en scenariogewichten.

---

## Onderzoeksvragen (2–4 vragen, ~200 woorden)

1. **Hoe kan de discrete grid search-optimalisatie in de Responsible Business Simulator worden vervangen door een continue optimalisatiemethode, en welke methode (SLSQP, basin-hopping, genetisch algoritme) levert de beste balans tussen oplossingskwaliteit en rekentijd?**

2. **Wat zijn de wiskundige eigenschappen (convexiteit, multimodaliteit) van de geaggregeerde doelfunctie die ontstaat uit de gewogen combinatie van lineaire en sinusoïdale waarderingsfuncties, en welke implicaties heeft dit voor de keuze van optimalisatiemethode?**

3. **Hoe gevoelig is de optimale kapitaalallocatie voor veranderingen in scenariogewichten, KPI-gewichten en appreciatiefunctieparameters, en welke inzichten levert een formele gevoeligheidsanalyse (schaduwprijzen, parametrische analyse) op voor de besluitvormer?**

4. **In hoeverre leidt continue optimalisatie tot aantoonbaar betere allocaties (hogere totale appreciatie) vergeleken met de huidige grid search-methode, en wat is de marginale waarde van fijnere optimalisatie voor praktische besluitvorming?**

---

## Literatuur (3–5 kernreferenties)

1. **Pedersen, L.H., Fitzgibbons, S. & Pomorski, L. (2021).** Responsible Investing: The ESG-Efficient Frontier. *Journal of Financial Economics*, 142(2), 572–597.
   — *Theoretisch fundament voor het ESG-efficiënte frontier; toont aan dat ESG-restricties de portefeuilleoptimalisatie fundamenteel veranderen en introduceert het concept van multi-dimensionale efficiënte grenzen.*

2. **Greco, S., Ehrgott, M. & Figueira, J. (Eds.) (2016).** Multiple Criteria Decision Analysis: State of the Art Surveys. *Springer International Handbooks of OR/MS*, 2e druk.
   — *Standaardwerk voor MCDA; bevat de theoretische grondslagen van waarderingsfuncties, gewichtsmethoden en de overgang van discrete alternatieven naar continue optimalisatie.*

3. **Nocedal, J. & Wright, S.J. (2006).** Numerical Optimization. *Springer Series in Operations Research*, 2e druk.
   — *Referentiewerk voor de optimalisatiemethoden (SQP, interior point, KKT-analyse) die in dit onderzoek worden toegepast.*

4. **Ben-Tal, A., El Ghaoui, L. & Nemirovski, A. (2009).** Robust Optimization. *Princeton University Press*.
   — *Theoretisch kader voor robuuste optimalisatie onder parameteronzekerheid; basis voor de scenariorobuustheidsanalyse.*

5. **Berg, F., Koelbel, J.F. & Rigobon, R. (2022).** Aggregate Confusion: The Divergence of ESG Ratings. *Review of Finance*, 26(6), 1315–1344.
   — *Motiveert waarom robuustheidsmethoden noodzakelijk zijn bij multi-criteria beslissingen op basis van ESG-data; toont significante divergentie tussen ratingbureaus aan.*

---

## Data (~200 woorden)

Dit onderzoek maakt primair gebruik van de vijf demonstratiecasussen die zijn opgenomen in het open-source Vlinder/tRBS-pakket: Beerwiser (waterrecycling vs. veiligheidstraining bij een bierproducent), Refugee (integratiebeleid voor vluchtelingen), DSM (energiebronkeuze), IZZ (personeelsontwikkeling in de zorg) en NEMO (dakrenovatie museum). Elk van deze casussen bevat een volledig gespecificeerde inputstructuur: investeringsopties met bijbehorende interne variabelen, scenario's met externe variabelen en gewichten, vaste inputs, KPI-definities met waarderingsfuncties en gewichten, en een afhankelijkhedengraaf met berekeningsregels. De omvang varieert van 2 tot 4 investeringsopties, 2 tot 3 scenario's, en 3 tot 6 KPI's per casus.

Daarnaast wordt, in samenwerking met PwC, ten minste één realistische praktijkcasus geconstrueerd op basis van geanonimiseerde klantdata. Deze casus zal een hogere dimensionaliteit hebben (meer investeringsopties en KPI's) om de schaalbaarheid van de optimalisatiemethoden te testen.

Alle data zijn direct beschikbaar: de demoscenario's zijn opgenomen in de publiek toegankelijke GitHub-repository van het Vlinder-pakket (Excel/CSV/JSON-formaat). De PwC-casus wordt intern beschikbaar gesteld via het tRBS-projectteam. Er hoeven geen externe data verzameld te worden.

---

## Methodologie (500–1000 woorden)

Dit onderzoek ontwikkelt een continue optimalisatiemethode voor kapitaalallocatie binnen het bestaande raamwerk van de Responsible Business Simulator (tRBS/Vlinder). De methodologie bestaat uit vier onderdelen: (i) wiskundige formulering van het optimalisatieprobleem, (ii) analyse van de doelfunctie-eigenschappen, (iii) implementatie en vergelijking van oplossingsmethoden, en (iv) gevoeligheids- en robuustheidsanalyse.

### i. Wiskundige formulering

Het allocatieprobleem wordt geformuleerd als een niet-lineair programmeringsprobleem (NLP). Laat **x** = (x_1, ..., x_k) de verdeling van budget over k investeringsopties zijn. De doelfunctie is de gewogen totale appreciatie zoals berekend door tRBS:

    max F(x) = Sum_s w_s * Sum_i w_i * v_i(g_i(x, s_s))

onder de restricties: Sum_j x_j = B (budgetrestrictie), x_j >= 0 voor alle j,

waarbij w_s de scenariogewichten zijn, w_i de genormaliseerde KPI-gewichten (twee-laags: KPI-gewicht x themagewicht), v_i de appreciatiefunctie (lineair of sinusoïdaal) voor KPI i, g_i de simulatiefunctie die via het afhankelijkheidsgrafenmodel van tRBS de KPI-waarde berekent, en B het totale beschikbare budget. De beslissingsvariabelen liggen op een (k-1)-dimensionale simplex, een compacte convexe verzameling.

### ii. Analyse van de doelfunctie

Een kernbijdrage van dit onderzoek is de wiskundige karakterisering van de doelfunctie F(x). De tRBS gebruikt twee typen waarderingsfuncties: lineaire functies v(x) = (x - s)/(e - s) * 100 en sinusoïdale functies v(x) = sin(pi/2 * (x - s)/(e - s)) * 100. Lineaire waarderingsfuncties behouden de (mogelijke) convexiteit of concaviteit van de onderliggende simulatiefunctie g. Sinusoïdale functies zijn concaaf op het interval [s, e] (aangezien sin(.) concaaf is op [0, pi/2]), wat betekent dat de samengestelde functie v(g(x)) concaaf is wanneer g lineair of affien is in x — een eigenschap die volgt uit de compositieregel voor concave functies (Boyd & Vandenberghe, 2004, par. 3.2.4).

Wanneer g niet-lineair is (bijvoorbeeld door multiplicatieve afhankelijkheden in het tRBS-model), kan F(x) niet-convex en multimodaal worden. Dit onderzoek brengt voor de bestaande demoscenario's (Beerwiser, Refugee, DSM) empirisch in kaart of en waar multimodaliteit optreedt, door het optimalisatielandschap te visualiseren (voor k = 2) en door het aantal lokale optima te tellen via multi-start experimenten (voor k > 2).

### iii. Oplossingsmethoden en benchmarking

Er worden drie klassen van optimalisatiemethoden geïmplementeerd en vergeleken:

**a) Gradiënt-gebaseerde lokale methoden:** Sequential Quadratic Programming (SLSQP) via scipy.optimize.minimize (Kraft, 1988; Nocedal & Wright, 2006). SLSQP lost iteratief kwadratische deelproblemen op en convergeert kwadratisch nabij een optimum. De methode is geschikt voor de simplexrestrictie en levert schaduwprijzen (duale variabelen) als bijproduct. Om het risico van lokale optima te beperken, wordt een multi-start strategie toegepast met N = 100–1000 willekeurige startpunten op de simplex (uniform bemonsterd via de Dirichlet-verdeling).

**b) Globale optimalisatie:** Basin-hopping (Wales & Doye, 1997), beschikbaar via scipy.optimize.basinhopping, combineert lokale optimalisatie met stochastische perturbaties om het landschap breder te verkennen. Deze methode is bijzonder geschikt voor matig multimodale problemen en biedt een betere balans tussen exploratie en exploitatie dan pure multi-start.

**c) Metaheuristieken:** Een genetisch algoritme (GA) met continue representatie, geïnspireerd door NSGA-II (Deb et al., 2002), wordt geïmplementeerd als referentiemethode. Hoewel GA's doorgaans minder efficiënt zijn dan gradiënt-gebaseerde methoden voor gladde problemen (Rios & Sahinidis, 2013), bieden zij het voordeel dat geen gradiëntinformatie nodig is en dat zij van nature meerdere oplossingen parallel evalueren.

**d) Benchmark tegen grid search:** Alle methoden worden vergeleken met de bestaande grid search van tRBS op drie criteria: (1) oplossingskwaliteit (bereikte totale appreciatie), (2) rekentijd, en (3) schaalbaarheid bij toenemend aantal investeringsopties k. De vergelijking vindt plaats op de vijf bestaande demoscenario's in tRBS.

### iv. Gevoeligheids- en robuustheidsanalyse

Na het vinden van de optimale allocatie x* wordt een uitgebreide gevoeligheidsanalyse uitgevoerd:

- **Schaduwprijzen:** De duale variabele op de budgetrestrictie geeft de marginale waarde van een extra eenheid budget. Dit volgt direct uit de KKT-condities van het NLP (Nocedal & Wright, 2006, Hfdst. 12).

- **Parametrische gevoeligheid:** Via de impliciete-functiestelling op het KKT-systeem (Fiacco, 1983) wordt berekend hoe x* verschuift bij veranderingen in scenariogewichten, KPI-gewichten en appreciatiefunctieparameters. Dit levert een Jacobi-matrix op die de besluitvormer inzicht geeft in welke aannames het meest bepalend zijn voor de aanbevolen allocatie.

- **Scenariorobuustheid:** Door de optimalisatie te herhalen voor variaties in scenariogewichten wordt een robuustheidsband rondom de optimale allocatie geconstrueerd. Hierbij wordt de methodologie van Bertsimas & Sim (2004) gevolgd, die de "prijs van robuustheid" kwantificeert — het verlies aan optimale appreciatie dat nodig is om een allocatie te vinden die goed presteert onder alle plausibele scenariogewichten.

Alle methoden worden geïmplementeerd in Python en geïntegreerd in de bestaande tRBS-codebase als uitbreiding van de optimize-module. De code wordt beschikbaar gesteld als open-source bijdrage aan het Vlinder-pakket.

---

## Onderzoeksnotities

### Strategische overwegingen voor een 8.0+
- **Onderzoeksvraag 2** (wiskundige eigenschappen) toont formele rigour — niet simpelweg "scipy inpluggen"
- **Onderzoeksvraag 3** (gevoeligheidsanalyse) voegt echte beslissingsondersteuningswaarde toe
- **Onderzoeksvraag 4** (benchmarking) biedt de empirische ruggengraat

### Nog te bespreken
- Titel: de huidige werktitel is lang — kortere versie gewenst?
- Scope: 4 onderzoeksvragen is ambitieus — overwegen om vraag 4 in vraag 1 te integreren?
- PwC-casus: is er al een concrete casus beschikbaar?
