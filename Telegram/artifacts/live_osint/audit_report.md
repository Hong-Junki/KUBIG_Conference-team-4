# Live OSINT Feed Audit Report

- Generated at: `2026-07-04T16:11:51.954244+00:00`
- Freshness threshold: `30` days
- Raw messages: `806`
- Extracted rows: `806`
- Conflict events: `213`
- Fresh conflict events: `78`
- Non-conflict extracted rows: `593`

## Channel Detection Rates

| channel | raw_messages | conflict_events | detection_rate |
| --- | --- | --- | --- |
| KyivIndependent_official | 253 | 82 | 32.4% |
| OSINTdefender | 196 | 40 | 20.4% |
| MiddleEastEye_TG | 147 | 39 | 26.5% |
| liveuamap | 109 | 40 | 36.7% |
| aljazeeraenglish | 50 | 5 | 10.0% |
| bellingcat_en | 50 | 7 | 14.0% |
| demo_security | 1 | 0 | 0.0% |

## Quality Flags

| flag | count | rate_among_conflict_events |
| --- | --- | --- |
| missing_country | 17 | 8.0% |
| missing_location | 17 | 8.0% |
| missing_coordinates | 17 | 8.0% |
| country_centroid_fallback | 111 | 52.1% |
| low_confidence_lt_0.75 | 6 | 2.8% |
| high_severity_low_confidence | 5 | 2.3% |
| negation_or_uncertainty_suspects | 3 | 1.4% |
| older_than_30_days | 135 | 63.4% |
| possible_duplicate_groups | 3 | - |

## Top Matched Keywords

| keyword | count |
| --- | --- |
| drone | 72 |
| attack | 59 |
| killed | 51 |
| missile | 50 |
| wounded | 19 |
| protest | 7 |
| airstrike | 6 |
| casualties | 5 |
| evacuation | 5 |
| troop | 4 |
| shelling | 4 |
| rocket | 4 |
| battle | 2 |
| clash | 1 |
| air strike | 1 |

## Samples To Review

### Missing Country

- `liveuamap` `2026-06-24T08:56:26+00:00` NO_COUNTRY / strike conf=0.75, sev=0.75: Commander of Unmanned systems of Ukrainian Armed forces confirmed drone strikes at the Sevastopol power substation https://liveuamap.com/en/2026/24-june-07-commander-of-unmanned... https://t.me/liveuamap/12179
- `KyivIndependent_official` `2026-06-23T14:15:55+00:00` NO_COUNTRY / strike conf=0.77, sev=0.98: ⚡️Update: Death toll rises to 3, with 25 injured in Kryvyi Rih strike. At least three people were killed and 25 injured after a Russian Iskander-M ballistic missile with cluster... https://t.me/KyivIndependent_official/53848
- `OSINTdefender` `2026-06-21T19:47:33+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China has constructed full-scale replicas of the USS Gerald R. Ford aircraft carrier and an Arleigh Burke-class destroyer in the Taklamakan Desert for military training p... https://t.me/OSINTdefender/19296
- `KyivIndependent_official` `2026-06-21T16:46:27+00:00` NO_COUNTRY / civil_unrest conf=0.77, sev=0.35: ⚡️Former Polish lawmaker returns state honor in protest over Zelensky's award revocation. Former Polish lawmaker Piotr Fogler said on June 20 that he had returned his state hono... https://t.me/KyivIndependent_official/53816
- `OSINTdefender` `2026-06-17T04:18:54+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #USA Navigational warnings indicate a missile test in the Pacific, south of Kwajalein, scheduled for June 27-29, 2026, with a range of approximately 3,800 km. This test is likel... https://t.me/OSINTdefender/19248
- `MiddleEastEye_TG` `2026-06-12T11:31:32+00:00` NO_COUNTRY / strike conf=0.76, sev=0.75: 🎥 Drone footage from Wednesday showed a vehicle set on fire as police blocked a road to stop anti-immigrant violence for the second night in Belfast, Northern Ireland. A wave of... https://t.me/MiddleEastEye_TG/21930
- `KyivIndependent_official` `2026-06-03T13:35:31+00:00` NO_COUNTRY / strike conf=0.77, sev=0.90: ⚡️ Zelensky threatens to fire officials for delays in Patriot missile supplies. “Unfortunately, as of today, even the legal groundwork for this contract has yet to be completed,... https://t.me/KyivIndependent_official/53475
- `OSINTdefender` `2026-05-31T11:36:49+00:00` NO_COUNTRY / military_movement conf=0.72, sev=0.45: #USA #EU The Pentagon plans to reduce U.S. military capabilities in Europe, bringing troop levels back to 2021 levels as part of a strategy to encourage European allies to take... https://t.me/OSINTdefender/19076

### Missing Coordinates

- `liveuamap` `2026-06-24T08:56:26+00:00` NO_COUNTRY / strike conf=0.75, sev=0.75: Commander of Unmanned systems of Ukrainian Armed forces confirmed drone strikes at the Sevastopol power substation https://liveuamap.com/en/2026/24-june-07-commander-of-unmanned... https://t.me/liveuamap/12179
- `KyivIndependent_official` `2026-06-23T14:15:55+00:00` NO_COUNTRY / strike conf=0.77, sev=0.98: ⚡️Update: Death toll rises to 3, with 25 injured in Kryvyi Rih strike. At least three people were killed and 25 injured after a Russian Iskander-M ballistic missile with cluster... https://t.me/KyivIndependent_official/53848
- `OSINTdefender` `2026-06-21T19:47:33+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China has constructed full-scale replicas of the USS Gerald R. Ford aircraft carrier and an Arleigh Burke-class destroyer in the Taklamakan Desert for military training p... https://t.me/OSINTdefender/19296
- `KyivIndependent_official` `2026-06-21T16:46:27+00:00` NO_COUNTRY / civil_unrest conf=0.77, sev=0.35: ⚡️Former Polish lawmaker returns state honor in protest over Zelensky's award revocation. Former Polish lawmaker Piotr Fogler said on June 20 that he had returned his state hono... https://t.me/KyivIndependent_official/53816
- `OSINTdefender` `2026-06-17T04:18:54+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #USA Navigational warnings indicate a missile test in the Pacific, south of Kwajalein, scheduled for June 27-29, 2026, with a range of approximately 3,800 km. This test is likel... https://t.me/OSINTdefender/19248
- `MiddleEastEye_TG` `2026-06-12T11:31:32+00:00` NO_COUNTRY / strike conf=0.76, sev=0.75: 🎥 Drone footage from Wednesday showed a vehicle set on fire as police blocked a road to stop anti-immigrant violence for the second night in Belfast, Northern Ireland. A wave of... https://t.me/MiddleEastEye_TG/21930
- `KyivIndependent_official` `2026-06-03T13:35:31+00:00` NO_COUNTRY / strike conf=0.77, sev=0.90: ⚡️ Zelensky threatens to fire officials for delays in Patriot missile supplies. “Unfortunately, as of today, even the legal groundwork for this contract has yet to be completed,... https://t.me/KyivIndependent_official/53475
- `OSINTdefender` `2026-05-31T11:36:49+00:00` NO_COUNTRY / military_movement conf=0.72, sev=0.45: #USA #EU The Pentagon plans to reduce U.S. military capabilities in Europe, bringing troop levels back to 2021 levels as part of a strategy to encourage European allies to take... https://t.me/OSINTdefender/19076

### Low Confidence

- `OSINTdefender` `2026-06-21T19:47:33+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China has constructed full-scale replicas of the USS Gerald R. Ford aircraft carrier and an Arleigh Burke-class destroyer in the Taklamakan Desert for military training p... https://t.me/OSINTdefender/19296
- `OSINTdefender` `2026-06-17T04:18:54+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #USA Navigational warnings indicate a missile test in the Pacific, south of Kwajalein, scheduled for June 27-29, 2026, with a range of approximately 3,800 km. This test is likel... https://t.me/OSINTdefender/19248
- `OSINTdefender` `2026-05-31T11:36:49+00:00` NO_COUNTRY / military_movement conf=0.72, sev=0.45: #USA #EU The Pentagon plans to reduce U.S. military capabilities in Europe, bringing troop levels back to 2021 levels as part of a strategy to encourage European allies to take... https://t.me/OSINTdefender/19076
- `OSINTdefender` `2026-05-30T05:38:03+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China is constructing a vast military complex with over 80 launch pads, bunkers, and communication nodes near its nuclear missile silos to enhance its second-strike capab... https://t.me/OSINTdefender/19060
- `OSINTdefender` `2026-05-29T05:42:46+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China is deploying multiple warships, including Type 056A guided-missile frigates Wuzhou and Tianmen, along with a Type 054A guided-missile frigate Dali, to conduct close... https://t.me/OSINTdefender/19046
- `OSINTdefender` `2026-05-27T06:13:26+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #NK North Korea has tested a multi-caliber rocket launcher capable of firing the Hwasong-11Ra ballistic missile and guided 240mm rockets, as well as a tactical cruise missile la... https://t.me/OSINTdefender/19030

### Negation Or Uncertainty Suspects

- `OSINTdefender` `2026-06-18T17:20:41+00:00` UKR / strike conf=0.92, sev=0.75: #Russia #Ukraine #Belarus Will the incident with the bus in the Bryansk region be used for a new round of escalation? Russian and Belarusian officials officially blamed Ukrainia... https://t.me/OSINTdefender/19262
- `OSINTdefender` `2026-06-18T04:31:32+00:00` UKR / conflict_signal conf=0.77, sev=0.37: #Russia #Ukraine Dozens of Ukrainian drones attacked the Moscow Refinery, causing damage but no casualties, according to Mayor Sergey Sobyanin. Emergency services are currently... https://t.me/OSINTdefender/19257
- `KyivIndependent_official` `2026-05-30T19:56:02+00:00` UKR / strike conf=0.97, sev=0.75: ⚡️Kyiv denies Russia's claims of Ukrainian drone strike on Zaporizhzhia nuclear plant. "The version promoted by Russia does not withstand any verification of the facts," the Ukr... https://t.me/KyivIndependent_official/53416

### Older Than 30 Days

- `KyivIndependent_official` `2026-06-04T10:08:43+00:00` UKR / strike conf=0.97, sev=0.90: Why Ukraine is talking about ending ‘hot phase’ of Russia’s war before winter Kyiv hopes U.S.-mediated peace talks with Moscow could yield results before winter as Ukrainian off... https://t.me/KyivIndependent_official/53493
- `MiddleEastEye_TG` `2026-06-04T09:30:27+00:00` PSE / conflict_signal conf=0.96, sev=0.78: 📹 Activists erected a statue of Marwan Barghouti in London’s Parliament Square, before it was removed by police. Marwan Barghouti is a Palestinian political leader who has been... https://t.me/MiddleEastEye_TG/21888
- `KyivIndependent_official` `2026-06-04T08:27:32+00:00` UKR / strike conf=0.97, sev=0.83: ⚡️ 16 killed, 86 injured in Russian attacks across Ukraine over past day, massive drone assault hits Kherson. The Air Force said Russia launched 293 drones, 264 of which were in... https://t.me/KyivIndependent_official/53492
- `MiddleEastEye_TG` `2026-06-04T07:31:16+00:00` LBN / military_movement conf=0.96, sev=0.40: 📹 Last week, Israeli forces ordered the evacuation of Tyre, a city in southern Lebanon with major historical and cultural significance, and declared all areas south of the Zahra... https://t.me/MiddleEastEye_TG/21886
- `OSINTdefender` `2026-06-04T05:53:26+00:00` USA / strike conf=0.92, sev=0.90: #USA #Iran U.S. fighters fired a missile at a Botswana-flagged tanker heading for Iran's Kharg terminal, destroying its engine room and immobilizing it due to defiance of the ma... https://t.me/OSINTdefender/19110
- `MiddleEastEye_TG` `2026-06-04T05:02:45+00:00` PSE / conflict_signal conf=0.96, sev=0.60: Progressive candidate Adam Hamawy has won the Democratic primary for New Jersey’s 12th Congressional District. Hamawy, a US Army veteran and surgeon who volunteered in Gaza duri... https://t.me/MiddleEastEye_TG/21884
- `KyivIndependent_official` `2026-06-04T04:53:14+00:00` UKR / strike conf=0.97, sev=0.75: ⚡️Russian drone attack targets infrastructure in Kyiv suburb, injuring 1. A Russian drone struck infrastructure in the Boryspil region of Kyiv Oblast overnight on June 4, sparki... https://t.me/KyivIndependent_official/53488
- `MiddleEastEye_TG` `2026-06-03T22:03:32+00:00` KWT / strike conf=0.96, sev=0.90: Iran on Wednesday said that the strikes on Kuwait's airport that caused extensive damage were the result of a US Patriot missile interceptor hit, a claim that US Central Command... https://t.me/MiddleEastEye_TG/21882

### Possible Duplicate Groups

- Group 1: 2 events
  - `OSINTdefender` `2026-06-22T18:09:00+00:00` UKR / conflict_signal conf=0.92, sev=0.75: #Russia #Ukraine On June 22, 2026, Ukrainian Storm Shadow cruise missiles struck the Voronezh Semiconductor Devices Plant, causing extensive damage, a major fire, and reportedly... https://t.me/OSINTdefender/19305
  - `KyivIndependent_official` `2026-06-22T12:33:09+00:00` UKR / conflict_signal conf=0.97, sev=0.75: ⚡️ Russian attack damages production facility of Ukraine’s FPV giant. “This is war. We were prepared for such events,” the company’s founder, Yaroslav Gryshyn, said. “The enemy... https://t.me/KyivIndependent_official/53828
- Group 2: 2 events
  - `OSINTdefender` `2026-05-29T17:55:26+00:00` ROU / strike conf=0.92, sev=0.75: #Germany #Romania #Russia German Chancellor Friedrich Merz emphasized that Germany is prepared to defend the territory of NATO allies, particularly in light of recent security c... https://t.me/OSINTdefender/19056
  - `KyivIndependent_official` `2026-05-29T12:58:59+00:00` ROU / strike conf=0.97, sev=0.75: ⚡️NATO condemns ‘reckless’ Russian drone strike on Romanian residential building NATO Secretary General Mark Rutte has said "Russia's reckless behaviour is a danger to us all,"... https://t.me/KyivIndependent_official/53396
- Group 3: 2 events
  - `OSINTdefender` `2026-05-29T05:08:37+00:00` ROU / strike conf=0.92, sev=0.75: #Romania A Russian drone crashed in Galați, Romania, causing minor damage to a residential building and injuring two people. The drone was reported to have an explosive payload,... https://t.me/OSINTdefender/19045
  - `KyivIndependent_official` `2026-05-29T01:27:35+00:00` ROU / strike conf=0.97, sev=0.75: ⚡️Drone reportedly strikes residential building in Romania. A drone reportedly struck a residential building in Galati, Romania, overnight on May 29, news outlet Viata Libera re... https://t.me/KyivIndependent_official/53391

## Suggested Next Actions

- If missing coordinates are high, expand `CITY_COORDS` and country aliases in `src/live_osint/extraction.py`.
- If old events are high, add a time-window filter during export or collection.
- If low-confidence events are useful, lower display threshold; if noisy, raise it.
- If one keyword dominates false positives, reduce its weight or require a country/location match.
