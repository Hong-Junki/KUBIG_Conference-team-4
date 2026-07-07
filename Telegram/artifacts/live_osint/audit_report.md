# Live OSINT Feed Audit Report

- Generated at: `2026-07-07T12:41:17.752236+00:00`
- Freshness threshold: `30` days
- Raw messages: `1143`
- Extracted rows: `1143`
- Conflict events: `320`
- Fresh conflict events: `182`
- Non-conflict extracted rows: `823`

## Channel Detection Rates

| channel | raw_messages | conflict_events | detection_rate |
| --- | --- | --- | --- |
| KyivIndependent_official | 387 | 137 | 35.4% |
| OSINTdefender | 284 | 59 | 20.8% |
| MiddleEastEye_TG | 233 | 61 | 26.2% |
| liveuamap | 138 | 51 | 37.0% |
| aljazeeraenglish | 50 | 5 | 10.0% |
| bellingcat_en | 50 | 7 | 14.0% |
| demo_security | 1 | 0 | 0.0% |

## Quality Flags

| flag | count | rate_among_conflict_events |
| --- | --- | --- |
| missing_country | 26 | 8.1% |
| missing_location | 26 | 8.1% |
| missing_coordinates | 26 | 8.1% |
| country_centroid_fallback | 156 | 48.8% |
| low_confidence_lt_0.75 | 11 | 3.4% |
| high_severity_low_confidence | 9 | 2.8% |
| negation_or_uncertainty_suspects | 3 | 0.9% |
| older_than_30_days | 138 | 43.1% |
| possible_duplicate_groups | 5 | - |

## Top Matched Keywords

| keyword | count |
| --- | --- |
| attack | 116 |
| drone | 95 |
| killed | 81 |
| missile | 71 |
| wounded | 26 |
| airstrike | 11 |
| protest | 9 |
| casualties | 9 |
| explosion | 7 |
| battle | 6 |
| troop | 6 |
| rocket | 6 |
| evacuation | 5 |
| shelling | 4 |
| clash | 1 |

## Samples To Review

### Missing Country

- `OSINTdefender` `2026-07-06T10:52:49+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China's test-fire of a long-range missile from a nuclear-powered submarine into the Pacific Ocean occurred on July 6, 2026, and has drawn significant criticism from regio... https://t.me/OSINTdefender/19392
- `OSINTdefender` `2026-07-03T08:52:37+00:00` NO_COUNTRY / military_movement conf=0.72, sev=0.45: #EU #USA Marco Rubio reportedly helped block Defense Secretary Pete Hegseth's plan to announce significant troop cuts in Europe during a NATO meeting last month. Subscribe to @O... https://t.me/OSINTdefender/19373
- `MiddleEastEye_TG` `2026-07-03T03:04:03+00:00` NO_COUNTRY / conflict_signal conf=0.76, sev=0.75: An Israeli bill to restrict the adhan, the Muslim call to prayer, passed a preliminary reading in parliament on Wednesday. Palestinians condemned the legislation as a "declarati... https://t.me/MiddleEastEye_TG/22113
- `KyivIndependent_official` `2026-07-02T14:05:14+00:00` NO_COUNTRY / conflict_signal conf=0.77, sev=0.75: ⚡️ Around 800,000 of Ukrainian publisher’s books destroyed in overnight Russian attack. According to BookChef, the central warehouse of their logistics partner — where their boo... https://t.me/KyivIndependent_official/54035
- `KyivIndependent_official` `2026-07-02T02:23:27+00:00` NO_COUNTRY / strike conf=0.77, sev=0.83: ⚡️Update: 2 people have been killed and at least 20 others injured in the attack, while drone threats in the city remain ongoing. As of 5 a.m. local time, officials had also rec... https://t.me/KyivIndependent_official/54024
- `OSINTdefender` `2026-06-30T22:26:58+00:00` NO_COUNTRY / strike conf=0.72, sev=0.85: #China The People's Liberation Army Rocket Force (PLARF), formerly the Second Artillery Corps, was established on 1 July 1966, making it nearly 60 years old as of 2026. Subscrib... https://t.me/OSINTdefender/19362
- `OSINTdefender` `2026-06-30T18:15:01+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #USA Northrop Grumman has revealed photos of the LGM-35A Sentinel, a new-generation intercontinental ballistic missile designed to replace the aging Minuteman III. This missile... https://t.me/OSINTdefender/19360
- `KyivIndependent_official` `2026-06-29T21:57:30+00:00` NO_COUNTRY / shelling_explosion conf=0.77, sev=0.75: ⚡️Massive explosion in Monaco injures Ukrainian family, media reports. Authorities in Monaco said the blast was "likely an attack." The explosion injured a 13-year-old girl and... https://t.me/KyivIndependent_official/53978

### Missing Coordinates

- `OSINTdefender` `2026-07-06T10:52:49+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China's test-fire of a long-range missile from a nuclear-powered submarine into the Pacific Ocean occurred on July 6, 2026, and has drawn significant criticism from regio... https://t.me/OSINTdefender/19392
- `OSINTdefender` `2026-07-03T08:52:37+00:00` NO_COUNTRY / military_movement conf=0.72, sev=0.45: #EU #USA Marco Rubio reportedly helped block Defense Secretary Pete Hegseth's plan to announce significant troop cuts in Europe during a NATO meeting last month. Subscribe to @O... https://t.me/OSINTdefender/19373
- `MiddleEastEye_TG` `2026-07-03T03:04:03+00:00` NO_COUNTRY / conflict_signal conf=0.76, sev=0.75: An Israeli bill to restrict the adhan, the Muslim call to prayer, passed a preliminary reading in parliament on Wednesday. Palestinians condemned the legislation as a "declarati... https://t.me/MiddleEastEye_TG/22113
- `KyivIndependent_official` `2026-07-02T14:05:14+00:00` NO_COUNTRY / conflict_signal conf=0.77, sev=0.75: ⚡️ Around 800,000 of Ukrainian publisher’s books destroyed in overnight Russian attack. According to BookChef, the central warehouse of their logistics partner — where their boo... https://t.me/KyivIndependent_official/54035
- `KyivIndependent_official` `2026-07-02T02:23:27+00:00` NO_COUNTRY / strike conf=0.77, sev=0.83: ⚡️Update: 2 people have been killed and at least 20 others injured in the attack, while drone threats in the city remain ongoing. As of 5 a.m. local time, officials had also rec... https://t.me/KyivIndependent_official/54024
- `OSINTdefender` `2026-06-30T22:26:58+00:00` NO_COUNTRY / strike conf=0.72, sev=0.85: #China The People's Liberation Army Rocket Force (PLARF), formerly the Second Artillery Corps, was established on 1 July 1966, making it nearly 60 years old as of 2026. Subscrib... https://t.me/OSINTdefender/19362
- `OSINTdefender` `2026-06-30T18:15:01+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #USA Northrop Grumman has revealed photos of the LGM-35A Sentinel, a new-generation intercontinental ballistic missile designed to replace the aging Minuteman III. This missile... https://t.me/OSINTdefender/19360
- `KyivIndependent_official` `2026-06-29T21:57:30+00:00` NO_COUNTRY / shelling_explosion conf=0.77, sev=0.75: ⚡️Massive explosion in Monaco injures Ukrainian family, media reports. Authorities in Monaco said the blast was "likely an attack." The explosion injured a 13-year-old girl and... https://t.me/KyivIndependent_official/53978

### Low Confidence

- `OSINTdefender` `2026-07-06T10:52:49+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China's test-fire of a long-range missile from a nuclear-powered submarine into the Pacific Ocean occurred on July 6, 2026, and has drawn significant criticism from regio... https://t.me/OSINTdefender/19392
- `OSINTdefender` `2026-07-03T08:52:37+00:00` NO_COUNTRY / military_movement conf=0.72, sev=0.45: #EU #USA Marco Rubio reportedly helped block Defense Secretary Pete Hegseth's plan to announce significant troop cuts in Europe during a NATO meeting last month. Subscribe to @O... https://t.me/OSINTdefender/19373
- `OSINTdefender` `2026-06-30T22:26:58+00:00` NO_COUNTRY / strike conf=0.72, sev=0.85: #China The People's Liberation Army Rocket Force (PLARF), formerly the Second Artillery Corps, was established on 1 July 1966, making it nearly 60 years old as of 2026. Subscrib... https://t.me/OSINTdefender/19362
- `OSINTdefender` `2026-06-30T18:15:01+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #USA Northrop Grumman has revealed photos of the LGM-35A Sentinel, a new-generation intercontinental ballistic missile designed to replace the aging Minuteman III. This missile... https://t.me/OSINTdefender/19360
- `OSINTdefender` `2026-06-26T08:12:47+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #NK North Korea tested upgraded artillery and missile systems under Kim Jong Un, including an upgraded 240mm multiple rocket launcher with a 90-kilometer range and a tactical ba... https://t.me/OSINTdefender/19330
- `OSINTdefender` `2026-06-21T19:47:33+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China has constructed full-scale replicas of the USS Gerald R. Ford aircraft carrier and an Arleigh Burke-class destroyer in the Taklamakan Desert for military training p... https://t.me/OSINTdefender/19296
- `OSINTdefender` `2026-06-17T04:18:54+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #USA Navigational warnings indicate a missile test in the Pacific, south of Kwajalein, scheduled for June 27-29, 2026, with a range of approximately 3,800 km. This test is likel... https://t.me/OSINTdefender/19248
- `OSINTdefender` `2026-05-31T11:36:49+00:00` NO_COUNTRY / military_movement conf=0.72, sev=0.45: #USA #EU The Pentagon plans to reduce U.S. military capabilities in Europe, bringing troop levels back to 2021 levels as part of a strategy to encourage European allies to take... https://t.me/OSINTdefender/19076

### Negation Or Uncertainty Suspects

- `OSINTdefender` `2026-06-18T17:20:41+00:00` UKR / strike conf=0.92, sev=0.75: #Russia #Ukraine #Belarus Will the incident with the bus in the Bryansk region be used for a new round of escalation? Russian and Belarusian officials officially blamed Ukrainia... https://t.me/OSINTdefender/19262
- `OSINTdefender` `2026-06-18T04:31:32+00:00` UKR / conflict_signal conf=0.77, sev=0.37: #Russia #Ukraine Dozens of Ukrainian drones attacked the Moscow Refinery, causing damage but no casualties, according to Mayor Sergey Sobyanin. Emergency services are currently... https://t.me/OSINTdefender/19257
- `KyivIndependent_official` `2026-05-30T19:56:02+00:00` UKR / strike conf=0.97, sev=0.75: ⚡️Kyiv denies Russia's claims of Ukrainian drone strike on Zaporizhzhia nuclear plant. "The version promoted by Russia does not withstand any verification of the facts," the Ukr... https://t.me/KyivIndependent_official/53416

### Older Than 30 Days

- `liveuamap` `2026-06-06T08:40:30+00:00` LBN / conflict_signal conf=0.95, sev=0.78: Lebanon army says several soldiers including an officer have been killed in Israeli strike on their military vehicle along the Khardali-Nabatieh road in southern lebanon https:/... https://t.me/liveuamap/12137
- `liveuamap` `2026-06-05T06:50:04+00:00` LBN / conflict_signal conf=0.95, sev=0.78: Civil Defense: Seven people killed in Israeli airstrikes on Tyre, southern Lebanon, Thursday night/Friday morning https://lebanon.liveuamap.com/en/2026/5-june-06-civil-defense-s... https://t.me/liveuamap/12134
- `liveuamap` `2026-06-04T19:43:01+00:00` LBN / strike conf=0.95, sev=0.95: Ministry of Health: 5 dead and one wounded as a result of an airstrike on the town of Sahmar in the Western Bekaa https://lebanon.liveuamap.com/en/2026/4-june-19-ministry-of-hea... https://t.me/liveuamap/12133
- `KyivIndependent_official` `2026-06-04T10:08:43+00:00` UKR / strike conf=0.97, sev=0.90: Why Ukraine is talking about ending ‘hot phase’ of Russia’s war before winter Kyiv hopes U.S.-mediated peace talks with Moscow could yield results before winter as Ukrainian off... https://t.me/KyivIndependent_official/53493
- `MiddleEastEye_TG` `2026-06-04T09:30:27+00:00` PSE / conflict_signal conf=0.96, sev=0.78: 📹 Activists erected a statue of Marwan Barghouti in London’s Parliament Square, before it was removed by police. Marwan Barghouti is a Palestinian political leader who has been... https://t.me/MiddleEastEye_TG/21888
- `KyivIndependent_official` `2026-06-04T08:27:32+00:00` UKR / strike conf=0.97, sev=0.83: ⚡️ 16 killed, 86 injured in Russian attacks across Ukraine over past day, massive drone assault hits Kherson. The Air Force said Russia launched 293 drones, 264 of which were in... https://t.me/KyivIndependent_official/53492
- `MiddleEastEye_TG` `2026-06-04T07:31:16+00:00` LBN / military_movement conf=0.96, sev=0.40: 📹 Last week, Israeli forces ordered the evacuation of Tyre, a city in southern Lebanon with major historical and cultural significance, and declared all areas south of the Zahra... https://t.me/MiddleEastEye_TG/21886
- `OSINTdefender` `2026-06-04T05:53:26+00:00` USA / strike conf=0.92, sev=0.90: #USA #Iran U.S. fighters fired a missile at a Botswana-flagged tanker heading for Iran's Kharg terminal, destroying its engine room and immobilizing it due to defiance of the ma... https://t.me/OSINTdefender/19110

### Possible Duplicate Groups

- Group 1: 3 events
  - `OSINTdefender` `2026-07-03T17:45:11+00:00` UKR / conflict_signal conf=0.92, sev=0.75: #Ukraine #USA Zelensky emphasizes the need for Ukraine to establish its own production of Patriot missiles to enhance air defense capabilities following a significant Russian at... https://t.me/OSINTdefender/19379
  - `KyivIndependent_official` `2026-07-03T13:10:56+00:00` UKR / conflict_signal conf=0.97, sev=0.75: ⚡️Kyiv air quality plunges after Russia's biggest attack on capital of entire war. The smell of smoke and haze lingered over the capital. Authorities urge residents to keep wind... https://t.me/KyivIndependent_official/54053
  - `KyivIndependent_official` `2026-07-03T12:03:45+00:00` UKR / conflict_signal conf=0.97, sev=0.75: ⚡️Update: Monaco issues arrest warrant for Ukrainian woman suspected of bomb attack that injured Ukraine-born businessman. Monaco issued an Interpol "red notice" — an internatio... https://t.me/KyivIndependent_official/54052
- Group 2: 2 events
  - `OSINTdefender` `2026-06-28T05:22:35+00:00` USA / conflict_signal conf=0.92, sev=0.75: #Kuwait #Bahrain Iranian ballistic missiles and drones reportedly struck 8 US military targets, including Ali Al Salem Air Base in Kuwait and the US Navy's 5th Fleet base in Bah... https://t.me/OSINTdefender/19344
  - `liveuamap` `2026-06-28T02:56:07+00:00` USA / conflict_signal conf=0.95, sev=0.75: In a statement, Iran's Islamic Revolutionary Guard Corps (IRGC) has confirmed that it attacked US forces in two military bases in the Middle East, the Ali Al Salem air base in K... https://t.me/liveuamap/12192
- Group 3: 2 events
  - `OSINTdefender` `2026-06-22T18:09:00+00:00` UKR / conflict_signal conf=0.92, sev=0.75: #Russia #Ukraine On June 22, 2026, Ukrainian Storm Shadow cruise missiles struck the Voronezh Semiconductor Devices Plant, causing extensive damage, a major fire, and reportedly... https://t.me/OSINTdefender/19305
  - `KyivIndependent_official` `2026-06-22T12:33:09+00:00` UKR / conflict_signal conf=0.97, sev=0.75: ⚡️ Russian attack damages production facility of Ukraine’s FPV giant. “This is war. We were prepared for such events,” the company’s founder, Yaroslav Gryshyn, said. “The enemy... https://t.me/KyivIndependent_official/53828
- Group 4: 2 events
  - `OSINTdefender` `2026-05-29T17:55:26+00:00` ROU / strike conf=0.92, sev=0.75: #Germany #Romania #Russia German Chancellor Friedrich Merz emphasized that Germany is prepared to defend the territory of NATO allies, particularly in light of recent security c... https://t.me/OSINTdefender/19056
  - `KyivIndependent_official` `2026-05-29T12:58:59+00:00` ROU / strike conf=0.97, sev=0.75: ⚡️NATO condemns ‘reckless’ Russian drone strike on Romanian residential building NATO Secretary General Mark Rutte has said "Russia's reckless behaviour is a danger to us all,"... https://t.me/KyivIndependent_official/53396
- Group 5: 2 events
  - `OSINTdefender` `2026-05-29T05:08:37+00:00` ROU / strike conf=0.92, sev=0.75: #Romania A Russian drone crashed in Galați, Romania, causing minor damage to a residential building and injuring two people. The drone was reported to have an explosive payload,... https://t.me/OSINTdefender/19045
  - `KyivIndependent_official` `2026-05-29T01:27:35+00:00` ROU / strike conf=0.97, sev=0.75: ⚡️Drone reportedly strikes residential building in Romania. A drone reportedly struck a residential building in Galati, Romania, overnight on May 29, news outlet Viata Libera re... https://t.me/KyivIndependent_official/53391

## Suggested Next Actions

- If missing coordinates are high, expand `CITY_COORDS` and country aliases in `src/live_osint/extraction.py`.
- If old events are high, add a time-window filter during export or collection.
- If low-confidence events are useful, lower display threshold; if noisy, raise it.
- If one keyword dominates false positives, reduce its weight or require a country/location match.
