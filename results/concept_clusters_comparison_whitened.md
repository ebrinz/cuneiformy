# Concept Cluster Comparison: GloVe vs EmbeddingGemma

Reverse-query reading: English seed -> top-10 Sumerian nearest neighbors, then for each Sumerian word, top-5 English nearest neighbors in the same space.

Human-read qualitative gate for phase A. The goal is to judge which space produces more semantically coherent clusters for the concept domains in `docs/NEAR_TERM_STRATEGY.md`.

## Domain: creation

### `create`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **dim2** -> create, creating, created, make, could | **dim2** -> create, creating, creates, created, crea |
| 2 | **badim2** -> create, creating, make, build, bring | **badim2** -> create, creates, creating, created, crea |
| 3 | **badim2ma** -> create, make, could, can, would | **dim2dim2** -> create, creating, creates, created, crea |
| 4 | **badim2dim2maba** -> create, make, could, would, build | **mudim2dim2** -> create, creating, creates, created, crea |
| 5 | **badim2ta** -> create, make, could, bring, would | **xdim2** -> create, creating, creates, created, crea |
| 6 | **baabdim2** -> create, could, to, would, make | **badim2dim2maba** -> create, creates, creating, created, crea |
| 7 | **munadim2** -> create, build, make, creating, to | **badim2ma** -> create, creates, creating, created, crea |
| 8 | **xdim2** -> create, creating, created, build, designed | **mune2dim2** -> create, creating, creates, created, crea |
| 9 | **dim2dim2** -> create, creating, created, make, build | **dim2dim2ma** -> create, creating, creates, created, crea |
| 10 | **dim2e** -> create, rather, make, creating, can | **mudim2** -> create, creating, creates, created, crea |

### `begin`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **muunkur9** -> able, would, to, they, come | **simug** -> smith, smiths, kaunda, schmidt, schmidts |
| 2 | **muunku4** -> would, able, to, want, they | **simugta** -> smith, smiths, kaunda, schmidt, stsmith |
| 3 | **nutillede3** -> come, complete, but, would, so | **simugne** -> smith, smiths, kaunda, schmidt, silversmith |
| 4 | **baramuuntil** -> come, would, make, take, they | **simug2** -> smith, smiths, kaunda, schmidt, smithii |
| 5 | **i3til** -> would, come, not, should, but | **simuggal** -> smith, smiths, schmidt, kaunda, j.smith |
| 6 | **muniinkur9** -> enter, able, to, take, would | **simugkam** -> smith, smiths, birthday, schmidt, kaunda |
| 7 | **baankur9reen** -> able, could, would, enter, to | **simugke4** -> smith, smiths, kaunda, schmidt, deity |
| 8 | **baabtah** -> to, take, would, could, able | **simugneta** -> smith, smiths, kaunda, shepherd, schmidt |
| 9 | **muunte** -> come, they, would, to, take | **gal:simug** -> smith, smiths, big, x, biggest |
| 10 | **nutilledam** -> complete, would, come, not, will | **kas4sze3** -> runner, runners, smuggler, smugglers, correr |

### `birth`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **amaibila** -> mother, father, daughter, child, son | **namtilanisze3** -> life, lives, lifes, leben, live |
| 2 | **dumuka** -> child, mother, daughter, children, father | **namtilani** -> life, lives, lifes, leben, lebens |
| 3 | **dumumunusbi** -> child, mother, daughter, woman, wife | **namtil3laniše3** -> life, lives, lifes, live, leben |
| 4 | **amazimu** -> mother, child, daughter, wife, father | **namtizu** -> life, lifes, lives, leben, lif |
| 5 | **amaugu** -> mother, child, daughter, woman, her | **namtilana** -> life, lives, lifes, leben, lebens |
| 6 | **dumuki** -> child, mother, daughter, parents, father | **namtilazu** -> life, lives, lifes, leben, lebens |
| 7 | **dumuzu** -> child, mother, daughter, parents, father | **namtil3zu** -> life, lives, lifes, leben, lebens |
| 8 | **amaarhuš** -> mother, daughter, her, wife, father | **namtilasze3** -> life, lives, lifes, leben, live |
| 9 | **dumu_** -> child, mother, daughter, parents, father | **namtila** -> life, lives, lifes, leben, lebens |
| 10 | **amanita** -> mother, wife, daughter, father, husband | **namtimu** -> life, lifes, lives, lebens, lif |

### `origin`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **ankur2kur2** -> different, these, certain, same, example | **akita** -> source, place, sources, origin, origins |
| 2 | **ipqu2ša** -> example, same, means, name, ” | **akitaka** -> source, sources, origin, origins, field |
| 3 | **kur2kur2ra** -> different, these, same, rather, those | **akitakakam** -> source, sources, origin, origins, origination |
| 4 | **kur2kur2** -> different, these, certain, example, types | **garrani** -> place, places, spot, lugar, region |
| 5 | **t,a3** -> same, example, find, this, only | **garraam6** -> place, places, spot, lugar, region |
| 6 | **iniminim** -> word, what, same, one, even | **kita** -> place, places, spot, locus, loci |
| 7 | **kakibisze3** -> ., this, the, same, but | **aki** -> place, places, turn, source, locus |
| 8 | **szembi2zida** -> the, same, ., of, though | **jarrani** -> place, places, spot, region, side |
| 9 | **ipqu2** -> god, ”, hence, “, thus | **ŋarrani** -> place, places, spot, side, sides |
| 10 | **dijirja2** -> deity, god, gods, one, same | **garra** -> place, places, spot, lugar, plaats |

### `emerge`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **bi2inmu2mu2** -> grow, come, able, make, need | **muunzizi** -> rise, rises, increase, uprise, rising |
| 2 | **bi2inpa3** -> find, come, able, to, make | **muunzig3** -> rise, rises, raise, increase, uprise |
| 3 | **he2mu2mu2** -> grow, come, able, growing, could | **muniinkur9** -> enter, entered, enters, entering, entrar |
| 4 | **mu2mu2** -> grow, grown, growing, grows, come | **dulla** -> cover, covers, covering, covered, cubrir |
| 5 | **bi2mu2mu2** -> grow, come, able, growing, grown | **gaankur9** -> enter, entered, enters, entering, entrar |
| 6 | **mu2mu2mu2** -> grow, grown, growing, grows, come | **he2emdul** -> speak, spoken, speakes, speaks, know |
| 7 | **munamu2mu2** -> grow, come, growing, able, grown | **kur9** -> enter, entered, enters, entering, entrance |
| 8 | **baanpa3** -> find, come, take, able, turn | **baanziziia** -> rise, rises, increase, increases, rising |
| 9 | **numu2mu2** -> grow, grown, growing, grows, come | **muunzi** -> rise, rises, know, knows, uprise |
| 10 | **numuundaanzizi** -> come, even, able, rise, could | **baankur9** -> enter, entered, enters, entering, entrar |

### `form`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **uludin2** -> form, rather, even, can, so | **uludin2** -> form, forms, descriptor, descriptors, signifier |
| 2 | **uludin** -> form, make, can, rather, to | **uludin** -> form, forms, descriptor, leave, vegetable |
| 3 | **dim2bi** -> create, can, rather, could, so | **ulud** -> form, forms, distant, leave, x |
| 4 | **dim4maam3** -> same, rather, only, so, example | **uludinbiše3** -> form, leave, forms, eat, descriptor |
| 5 | **kaam3** -> only, same, one, not, because | **ulutim2bi** -> form, x, forms, different, xs |
| 6 | **uludinbiše3** -> make, to, take, would, if | **ulutim2ta** -> x, form, forms, xs, different |
| 7 | **mubi** -> same, this, but, one, so | **inimma** -> word, witness, witnesses, witnessing, witnessed |
| 8 | **dannaam3** -> only, could, would, even, same | **inim** -> word, name, speaking, morpheme, words |
| 9 | **ulud** -> rather, form, even, so, can | **ulutim2** -> x, form, xs, forms, tens |
| 10 | **laamuta** -> same, ., this, so, only | **inime3** -> word, leave, name, form, turn |

### `separate`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **didli** -> several, other, many, few, two | **surrabi** -> place, places, spot, mountain, surface |
| 2 | **didlibi** -> several, one, two, few, three | **surra** -> border, press, boundary, borders, verges |
| 3 | **kur2sze3** -> different, same, all, these, those | **sur** -> press, presses, presse, border, prensa |
| 4 | **didliene** -> several, one, many, other, some | **u3bi2insurra** -> place, leave, surface, spot, places |
| 5 | **limmu4** -> three, four, two, six, five | **balla2** -> hang, turn, hangs, turns, hanged |
| 6 | **didlizu** -> several, few, well, one, some | **surrata** -> place, verges, mountain, outside, places |
| 7 | **limmuba** -> three, two, four, six, five | **su3bi** -> distant, faraway, distance, distances, distantly |
| 8 | **didlia** -> several, few, many, other, some | **u3bi2ak** -> leave, speak, speaks, spoken, turn |
| 9 | **minabi** -> two, three, four, one, five | **kur2gin7** -> different, difference, differently, differences, distinct |
| 10 | **didlita** -> several, other, many, few, some | **nukur2ru** -> different, difference, differences, differs, differently |

## Domain: fate_meaning

### `fate`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **namtar** -> fate, life, how, perhaps, turn | **namtar** -> cut, cuts, fate, fates, destiny |
| 2 | **namtag2** -> even, nothing, so, what, mind | **namtag2** -> fate, touch, touching, destiny, touches |
| 3 | **namtarzu** -> even, so, how, life, indeed | **namta** -> fate, life, destiny, fates, lifes |
| 4 | **namtag** -> even, touch, life, never, mind | **namtag** -> touch, touching, touches, fate, destiny |
| 5 | **namtarju10** -> so, life, because, even, my | **namtarra** -> cut, cuts, fate, couper, cutt |
| 6 | **namta** -> life, fate, rest, god, mind | **namtarju10** -> cut, cuts, fate, life, cutt |
| 7 | **namte** -> life, indeed, lives, god, even | **namtarzu** -> cut, cuts, fate, profit-sharing, fates |
| 8 | **namtarrani** -> even, might, if, turn, perhaps | **namtaba** -> fate, destiny, life, fates, lifes |
| 9 | **namtae3** -> leave, leaving, stay, return, not | **namtagga** -> touch, touching, touches, touched, toucher |
| 10 | **namtarra** -> cut, even, perhaps, life, so | **namtaggani** -> touch, touching, touches, fate, death |

### `destiny`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **namtag2** -> even, nothing, so, what, mind | **namtag2** -> fate, touch, touching, destiny, touches |
| 2 | **namtag** -> even, touch, life, never, mind | **namta** -> fate, life, destiny, fates, lifes |
| 3 | **namzaqum** -> if, so, hand, because, instead | **namtar** -> cut, cuts, fate, fates, destiny |
| 4 | **namta** -> life, fate, rest, god, mind | **namtag** -> touch, touching, touches, fate, destiny |
| 5 | **namte** -> life, indeed, lives, god, even | **namtaba** -> fate, destiny, life, fates, lifes |
| 6 | **namazu** -> know, how, what, father, life | **namte** -> life, lifes, lives, lif, lebens |
| 7 | **namsun5na** -> god, come, indeed, nothing, good | **namtarra** -> cut, cuts, fate, couper, cutt |
| 8 | **namgala** -> love, wish, greatness, surely, realize | **namtaggani** -> touch, touching, touches, fate, death |
| 9 | **nammu2** -> grow, life, beyond, come, need | **nambi** -> cut, cuts, fate, fates, destiny |
| 10 | **namjuruc** -> all, not, same, if, only | **namtarju10** -> cut, cuts, fate, life, cutt |

### `purpose`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **kislahzu** -> same, this, so, kind, it | **a2ag2ga2** -> assignment, assignments, arm, carry, measure |
| 2 | **sikildu3a** -> build, kind, rather, instead, so | **gu3de2aasz2** -> pour, pours, poured, pouring, ran |
| 3 | **du3akam** -> build, make, need, even, so | **ku3gane2** -> metal, metals, silver, pure, silvers |
| 4 | **badim2sze3** -> create, make, kind, can, you | **ku3ge** -> metal, metals, priest, priests, pure |
| 5 | **du3abi** -> build, to, make, create, would | **ku3gepa3da** -> find, finds, metal, metals, ﬁnds |
| 6 | **miriinsa4** -> must, so, need, same, make | **kidde3** -> carry, right, carries, place, xs |
| 7 | **mune2dim2** -> create, make, help, would, could | **sig2** -> x, xs, ten, tens, 10 |
| 8 | **du3x** -> build, make, create, develop, built | **sigₓumbin** -> nail, nails, fingernail, toenail, fingernails |
| 9 | **du3ake4** -> build, make, create, well, own | **ag2bi** -> measure, measures, measurement, measuring, mesure |
| 10 | **du3.** -> build, make, well, more, so | **ag2ga2** -> measure, measures, place, mesure, places |

### `decree`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **namensi2** -> priesthood, ruler, not, return, given | **dikud** -> judge, cut, cuts, profit-sharing, adjudicated |
| 2 | **ensi2mah** -> ruler, father, priest, king, rulers | **diku5bi** -> judge, cut, cuts, judice, court |
| 3 | **ensi2kabi** -> ruler, rulers, rule, king, thus | **diku5bime** -> judge, cut, court, courts, arbitrator |
| 4 | **name3** -> leave, return, not, whether, leaving | **diku5še3** -> judge, cut, cuts, cutt, jobes |
| 5 | **ensi2kam** -> ruler, rulers, rule, king, kingdom | **ensi2kaše3** -> ruler, rulers, rulership, king, command |
| 6 | **lugalmah** -> king, queen, kingdom, ii, monarch | **ensi2** -> ruler, rulers, authority, overlord, ruled |
| 7 | **ensi2** -> ruler, king, rulers, rule, kingdom | **ensi2kabi** -> ruler, rulers, rulership, ruled, co-ruler |
| 8 | **dikud** -> court, because, that, judge, but | **ensi2ka** -> ruler, rulers, ruled, rulership, co-ruler |
| 9 | **ensi2.** -> ruler, rulers, king, kingdom, monarch | **diku5dani** -> cut, cuts, judge, cutt, cutfather |
| 10 | **ensi2kasze3** -> ruler, rulers, king, rule, monarch | **diku5gal** -> cut, judge, cuts, cutt, jobes |

### `name`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **inimbita** -> even, but, so, what, not | **muni** -> name, word, nouns, noun, place |
| 2 | **inimma** -> word, what, even, nothing, come | **mubisze3** -> name, place, stand, noun, nouns |
| 3 | **inimbe6** -> word, what, even, come, so | **muzu** -> know, name, knows, known, familiarity |
| 4 | **inimmusze3** -> what, so, but, even, come | **mumusze3** -> name, word, hand, noun, element |
| 5 | **iniminimma** -> word, what, even, because, but | **mubiim** -> name, namesnik, named, cuts, cut |
| 6 | **iniminimmasze3** -> word, what, but, even, because | **muniim** -> name, mother, man, property, woman |
| 7 | **inimmax** -> word, what, even, fact, but | **munizu** -> know, knows, name, known, find |
| 8 | **muni** -> because, so, if, find, them | **inimmusze3** -> word, name, morpheme, morphemes, words |
| 9 | **inimbi** -> word, but, what, even, not | **musze3** -> bird, birds, name, word, hand |
| 10 | **kiinimma** -> what, actually, because, kind, this | **munisze3** -> name, child, word, place, arm |

### `order`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **munaus2** -> to, should, take, must, make | **ensi2ra** -> ruler, rulers, king, ruled, leader |
| 2 | **tillaam3** -> complete, would, only, all, not | **ensi2kabi** -> ruler, rulers, rulership, ruled, co-ruler |
| 3 | **munaniku4** -> would, to, make, take, must | **ensi2ke4ne** -> ruler, rulers, ruled, rulership, leader |
| 4 | **baandaribi** -> would, take, to, could, able | **ensi2kake4ne** -> ruler, rulers, ruled, authority, command |
| 5 | **munanikuxkwu634** -> would, must, to, will, can | **ensi2me** -> ruler, rulers, overlord, overlords, officer |
| 6 | **baannacum2** -> give, take, would, to, not | **ensi2kam** -> ruler, rulers, rulership, ruled, co-ruler |
| 7 | **imde6am3** -> carry, would, take, will, could | **ensi2kake4** -> ruler, rulers, rulership, ruled, command |
| 8 | **iszkuune2a** -> which, where, same, ., the | **ensi2kata** -> ruler, rulers, command, rulership, regulating |
| 9 | **ib2uru4asze3** -> would, take, could, to, should | **ensi2ke4** -> ruler, rulers, rulership, leader, ruled |
| 10 | **du3dam** -> build, would, to, able, could | **ensi2kame** -> ruler, rulers, rulership, ruled, overlord |

## Domain: self_soul

### `self`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **ni2te** -> self, kind, mind, person, sort | **ni2bi** -> self, selves, self-identity, self-identification, self-identifies |
| 2 | **ni2x** -> self, kind, man, own, child | **ni2za** -> self, selves, yourself, self-identity, self-identification |
| 3 | **ni2bi** -> self, so, kind, even, rather | **ni2** -> self, selves, self-identity, self-identification, self-identify |
| 4 | **ni2** -> self, kind, so, ?, you | **ni2te** -> self, selves, self-identity, self-identification, self-identifies |
| 5 | **ni2teana** -> self, kind, person, woman, mind | **ni2zu** -> self, selves, self-identity, self-awareness, self-identification |
| 6 | **ni2e** -> self, child, own, kind, never | **ni2bia** -> self, selves, self-identity, self-identifies, yourself |
| 7 | **ni2teni** -> self, kind, own, mind, mother | **ni2x** -> self, selves, self-identity, self-identification, self-identifies |
| 8 | **ni2teani** -> self, woman, person, kind, man | **ni2bisze3** -> self, selves, self-identity, self-identifies, identity |
| 9 | **ni2za** -> self, kind, own, you, sort | **ni2teana** -> self, selves, self-identity, self-identifies, self-identification |
| 10 | **ni2tenaka** -> self, own, kind, once, what | **ni2bita** -> self, selves, self-identity, self-identification, self-identifies |

### `soul`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **nare** -> musician, singer, songwriter, artist, musicians | **namtil3zu** -> life, lives, lifes, leben, lebens |
| 2 | **en3duzu** -> you, so, always, actually, song | **namtil3bi** -> life, lives, lifes, leben, lebens |
| 3 | **szir3ra** -> song, songs, you, singing, love | **namtilazu** -> life, lives, lifes, leben, lebens |
| 4 | **galara** -> singer, song, love, musician, mother | **namtizu** -> life, lifes, lives, leben, lif |
| 5 | **naram6** -> musician, singer, who, music, artist | **namtilasze3** -> life, lives, lifes, leben, live |
| 6 | **en3dua** -> song, so, you, way, love | **namtila** -> life, lives, lifes, leben, lebens |
| 7 | **en3du** -> song, songs, album, love, one | **namtil3lazu** -> life, lives, lifes, leben, live |
| 8 | **en3dux** -> song, one, same, time, songs | **namtilaka** -> life, lives, lifes, leben, lebens |
| 9 | **sikillaza** -> kind, pure, you, always, sort | **namtilani** -> life, lives, lifes, leben, lebens |
| 10 | **balag** -> song, instrument, soul, album, music | **namtil3laka** -> life, lives, lifes, leben, live |

### `spirit`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **ŋa2tum3dug3ke4** -> bring, way, come, well, even | **namtil3bi** -> life, lives, lifes, leben, lebens |
| 2 | **namsag9ga** -> good, always, really, know, you | **namtil3** -> life, lives, lifes, leben, live |
| 3 | **namursagga2mu** -> man, hero, him, but, same | **namtil3ŋu10** -> life, lives, lifes, leben, lebens |
| 4 | **ŋa2tum3dug3** -> bring, good, well, way, come | **namtil3am3** -> life, lives, lifes, leben, live |
| 5 | **namursagga2zu** -> man, always, great, kind, him | **namtilasze3** -> life, lives, lifes, leben, live |
| 6 | **mahzu** -> great, kind, good, what, really | **diŋirbi** -> deity, deities, divinity, divinities, god |
| 7 | **ursag9ga** -> good, always, know, nothing, you | **namtila** -> life, lives, lifes, leben, lebens |
| 8 | **namsa2e** -> equal, should, respect, good, give | **namtil3lani** -> life, lives, lifes, leben, live |
| 9 | **namkalaggana** -> good, but, strong, always, indeed | **namtil3la** -> life, lives, lifes, leben, live |
| 10 | **lamsa2rake4** -> good, always, you, place, kind | **namtil3lake4** -> life, lives, lifes, leben, live |

### `mind`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **muzua** -> know, you, what, n't, come | **dur2** -> buttocks, ass, tush, derriere, arse |
| 2 | **muzuzu** -> know, what, n't, you, so | **kituszbi** -> sit, sits, sittin, place, sitt |
| 3 | **he2zuzu** -> know, n't, want, you, what | **kituszani** -> sit, sits, sittin, sitt, sitdown |
| 4 | **nuzuzu** -> know, n't, you, what, want | **kituszanita** -> sit, sits, sittin, sitt, sentado |
| 5 | **inimŋal2** -> even, come, so, what, kind | **kitusz** -> sit, sits, sittin, sitt, sitdown |
| 6 | **nuzua** -> know, n't, you, what, come | **tuszasze3** -> sit, sits, sitt, sitdown, sittin |
| 7 | **zuzua** -> know, what, you, n't, how | **dur2bita** -> buttocks, ass, derriere, arse, tush |
| 8 | **namtaggani** -> nothing, even, know, mind, my | **kituszni** -> sit, sits, sittin, sitt, sitdown |
| 9 | **ŋe26emeen** -> you, so, n't, even, just | **kidzu** -> know, knows, place, known, knew |
| 10 | **ge26emeen** -> you, n't, so, even, just | **xgar** -> place, places, spot, region, lugar |

### `heart`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **sza3bi** -> one, heart, same, well, the | **sza3ba** -> heart, field, feel, feels, sense |
| 2 | **sza3bia** -> just, so, where, because, one | **sza3bi** -> heart, field, heartfield, side, feeling |
| 3 | **sza3zu** -> because, so, you, what, know | **sza3bia** -> field, heart, side, place, outside |
| 4 | **sza3za** -> heart, so, because, like, you | **sza3be2** -> heart, field, speak, feeling, joy |
| 5 | **sza3be2** -> so, you, just, what, even | **sza3** -> field, heart, place, ﬁeld, location |
| 6 | **sza3ta** -> something, so, my, even, heart | **_sza3ba** -> field, heart, garden, heartfield, room |
| 7 | **sza3zua** -> so, because, you, what, well | **sza3bita** -> heart, field, entity, price, mountain |
| 8 | **gabazu** -> you, chest, just, so, n't | **sza3ab** -> heart, field, sheep, wool, place |
| 9 | **sza3bagu10** -> so, one, just, even, you | **cag4** -> x, xs, ten, tens, 10 |
| 10 | **sza3ab** -> so, you, heart, my, way | **sza3banagar** -> field, place, heart, carpenter, room |

### `breath`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **zapaaŋ2** -> you, so, 'll, really, n't | **zapaag2zu** -> measure, breath, measures, breaths, know |
| 2 | **zapaaŋ2ŋu10** -> you, so, something, even, n't | **ku6gu10** -> fish, fishes, breathe, eat, breath |
| 3 | **zapaaŋ2zu** -> you, so, n't, know, something | **zapaaŋ2zu** -> know, knows, known, breath, knew |
| 4 | **zapaag2zu** -> you, so, if, get, n't | **zapaaŋ2** -> breath, breaths, breathe, breathing, x |
| 5 | **zapaaj2** -> you, so, just, if, like | **zapaaŋ2bi** -> breath, turn, turns, x, self |
| 6 | **zapaaŋ2bi** -> you, so, just, kind, even | **zapaag2bi** -> leave, measure, breath, measures, wind |
| 7 | **zapaaj2zu** -> you, so, if, just, what | **zapaag2** -> measure, measures, breath, leave, wind |
| 8 | **zapaag2** -> mouth, you, breath, away, so | **zapaaŋ2ŋu10** -> breath, hair, breathe, find, nose |
| 9 | **zapaaj2bi** -> you, so, if, just, kind | **zapaaj2zu** -> know, knows, place, measure, breath |
| 10 | **ga14** -> mouth, hand, you, eyes, someone | **zapaaj2bi** -> measure, measures, breath, x, place |

### `shadow`

| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |
|---|---|---|
| 1 | **enlil2la2zi** -> come, even, same, god, . | **barrata** -> outside, exterior, outsides, exteriors, outer |
| 2 | **u4gin7** -> so, come, even, ., it | **iribarra** -> city, outside, citys, cities, metropolis |
| 3 | **enlil2la2sze3** -> you, come, instead, so, like | **barrame** -> outside, exterior, outsides, exteriors, outer |
| 4 | **ninlil2la2sze3** -> come, you, me, god, actually | **barbarra** -> outside, exterior, outsides, exteriors, ouside |
| 5 | **udgin7** -> sun, like, so, come, you | **barrana** -> outside, exterior, outsides, exteriors, outer |
| 6 | **ku10** -> dark, eyes, so, n't, look | **ŋissubi** -> shade, leave, good, raise, shine |
| 7 | **ebgalla** -> big, dark, like, one, only | **iribarre** -> outside, city, citys, outsides, cities |
| 8 | **iggalla** -> one, even, big, so, only | **sza3nesza4** -> joy, feelgood, feel-good, good, joyfulness |
| 9 | **enlil2la2** -> you, come, god, ?, even | **barbia** -> outside, exterior, outsides, exteriors, outer |
| 10 | **musz3mezu** -> so, even, what, same, come | **barrim4** -> outside, exterior, outsides, exteriors, ouside |
