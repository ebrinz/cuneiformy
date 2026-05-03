# Lens 1: English displacement

Anchor pairs ranked by cosine similarity between the source-language token's aligned (projected) vector and the English gloss's native target vector. Low cosine = translation that geometrically misses.

## Unfiltered (includes anchor-quality noise)

| Source | English | Cos | Conf | Src |
|---|---|---|---|---|
| rin2 | lord | -0.0877 | 1.0000 | ETCSL |
| bun2 | s | -0.0646 | 0.7333 | ETCSL |
| jizzal | ear | -0.0635 | 0.9500 | ePSD2 |
| zage3 | foremost | -0.0426 | 0.5600 | ePSD2 |
| sa3bi2tum | barkeeper | -0.0391 | 0.5500 | ePSD2 |
| en | priest | -0.0371 | 0.9500 | ePSD2 |
| igira2 | heron | -0.0345 | 1.0000 | ETCSL |
| du14 | combat | -0.0327 | 0.9500 | ePSD2 |
| agazi | loss | -0.0276 | 0.6300 | ePSD2 |
| eellu | song | -0.0275 | 0.6200 | ePSD2 |
| araah | storehouse | -0.0224 | 0.5500 | ePSD2 |
| ia3 | h | -0.0223 | 0.8750 | ETCSL |
| mutiin | male | -0.0214 | 0.5600 | ePSD2 |
| gu2zal | scoundrel | -0.0195 | 0.5500 | ePSD2 |
| naga | potash | -0.0193 | 0.9500 | ePSD2 |
| uk | you | -0.0184 | 0.6667 | ETCSL |
| egir2 | x | -0.0159 | 0.6667 | ETCSL |
| gu2tul2 | work | -0.0143 | 0.8800 | ePSD2 |
| anga | moreover | -0.0140 | 0.6700 | ePSD2 |
| alad | protective | -0.0128 | 1.0000 | ETCSL |
| sir | dense | -0.0113 | 0.6800 | ePSD2 |
| igigal2 | reciprocal | -0.0109 | 0.7200 | ePSD2 |
| bi2nitum | beam | -0.0109 | 0.6100 | ePSD2 |
| par4 | g | -0.0103 | 0.7941 | ETCSL |
| il5 | you | -0.0100 | 1.0000 | ETCSL |
| uddata | hereafter | -0.0054 | 0.6000 | ePSD2 |
| dala2 | x | -0.0052 | 1.0000 | ETCSL |
| lamma | protective | -0.0051 | 0.7526 | ETCSL |
| amaargi4 | reversion | -0.0014 | 0.5500 | ePSD2 |
| naaj2 | fate | -0.0012 | 0.7400 | ePSD2 |
| habrud | did | -0.0008 | 0.8000 | ETCSL |
| ke | you | 0.0013 | 0.7500 | ETCSL |
| ece2 | said | 0.0016 | 1.0000 | ETCSL |
| sza3si | rail | 0.0048 | 0.6300 | ePSD2 |
| usz11 | poison | 0.0065 | 0.8100 | ePSD2 |
| nagabi | potash | 0.0088 | 0.9500 | ePSD2 |
| muu2a | youth | 0.0092 | 0.5700 | ePSD2 |
| muru9gin7 | rainstorm | 0.0105 | 0.5600 | ePSD2 |
| isin | stalk | 0.0106 | 0.6600 | ePSD2 |
| ere | throttle | 0.0109 | 0.5700 | ePSD2 |
| kamma | instrument | 0.0112 | 0.5500 | ePSD2 |
| sipar3 | implement | 0.0126 | 0.5600 | ePSD2 |
| naamtura | illness | 0.0127 | 0.9100 | ePSD2 |
| duga | command | 0.0136 | 0.8500 | ePSD2 |
| nu11 | x | 0.0142 | 0.8485 | ETCSL |
| dim4ma | check | 0.0153 | 0.5600 | ePSD2 |
| arali | earth | 0.0172 | 0.5900 | ePSD2 |
| diirgaa | bond | 0.0173 | 0.5700 | ePSD2 |
| nig2ga2ni | thing | 0.0174 | 0.6000 | ePSD2 |
| enmen | out | 0.0177 | 0.6667 | ETCSL |


## Filtered (anchor-quality rules applied)

_Rules: english_len>2, english_not_numeric, english_not_in_junk_set, sumerian_len>=2, anchor_confidence>=0.5_

| Source | English | Cos | Conf | Src |
|---|---|---|---|---|
| rin2 | lord | -0.0877 | 1.0000 | ETCSL |
| jizzal | ear | -0.0635 | 0.9500 | ePSD2 |
| zage3 | foremost | -0.0426 | 0.5600 | ePSD2 |
| sa3bi2tum | barkeeper | -0.0391 | 0.5500 | ePSD2 |
| en | priest | -0.0371 | 0.9500 | ePSD2 |
| igira2 | heron | -0.0345 | 1.0000 | ETCSL |
| du14 | combat | -0.0327 | 0.9500 | ePSD2 |
| agazi | loss | -0.0276 | 0.6300 | ePSD2 |
| eellu | song | -0.0275 | 0.6200 | ePSD2 |
| araah | storehouse | -0.0224 | 0.5500 | ePSD2 |
| mutiin | male | -0.0214 | 0.5600 | ePSD2 |
| gu2zal | scoundrel | -0.0195 | 0.5500 | ePSD2 |
| naga | potash | -0.0193 | 0.9500 | ePSD2 |
| uk | you | -0.0184 | 0.6667 | ETCSL |
| gu2tul2 | work | -0.0143 | 0.8800 | ePSD2 |
| anga | moreover | -0.0140 | 0.6700 | ePSD2 |
| alad | protective | -0.0128 | 1.0000 | ETCSL |
| sir | dense | -0.0113 | 0.6800 | ePSD2 |
| igigal2 | reciprocal | -0.0109 | 0.7200 | ePSD2 |
| bi2nitum | beam | -0.0109 | 0.6100 | ePSD2 |
| il5 | you | -0.0100 | 1.0000 | ETCSL |
| uddata | hereafter | -0.0054 | 0.6000 | ePSD2 |
| lamma | protective | -0.0051 | 0.7526 | ETCSL |
| amaargi4 | reversion | -0.0014 | 0.5500 | ePSD2 |
| naaj2 | fate | -0.0012 | 0.7400 | ePSD2 |
| habrud | did | -0.0008 | 0.8000 | ETCSL |
| ke | you | 0.0013 | 0.7500 | ETCSL |
| ece2 | said | 0.0016 | 1.0000 | ETCSL |
| sza3si | rail | 0.0048 | 0.6300 | ePSD2 |
| usz11 | poison | 0.0065 | 0.8100 | ePSD2 |
| nagabi | potash | 0.0088 | 0.9500 | ePSD2 |
| muu2a | youth | 0.0092 | 0.5700 | ePSD2 |
| muru9gin7 | rainstorm | 0.0105 | 0.5600 | ePSD2 |
| isin | stalk | 0.0106 | 0.6600 | ePSD2 |
| ere | throttle | 0.0109 | 0.5700 | ePSD2 |
| kamma | instrument | 0.0112 | 0.5500 | ePSD2 |
| sipar3 | implement | 0.0126 | 0.5600 | ePSD2 |
| naamtura | illness | 0.0127 | 0.9100 | ePSD2 |
| duga | command | 0.0136 | 0.8500 | ePSD2 |
| dim4ma | check | 0.0153 | 0.5600 | ePSD2 |
| arali | earth | 0.0172 | 0.5900 | ePSD2 |
| diirgaa | bond | 0.0173 | 0.5700 | ePSD2 |
| nig2ga2ni | thing | 0.0174 | 0.6000 | ePSD2 |
| enmen | out | 0.0177 | 0.6667 | ETCSL |
| cu4 | like | 0.0179 | 0.5455 | ETCSL |
| gu2jiri3 | breach | 0.0187 | 0.5700 | ePSD2 |
| dim4 | are | 0.0193 | 0.5833 | ETCSL |
| umah | placed | 0.0194 | 0.7500 | ETCSL |
| iszme | stone | 0.0198 | 0.5800 | ePSD2 |
| nagata | potash | 0.0214 | 0.9500 | ePSD2 |

