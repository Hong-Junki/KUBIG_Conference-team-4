# Live OSINT Feed Audit Report

- Generated at: `2026-07-11T01:19:11.186129+00:00`
- Freshness threshold: `30` days
- Raw messages: `1318`
- Extracted rows: `1318`
- Conflict events: `358`
- Fresh conflict events: `212`
- Non-conflict extracted rows: `960`

## Channel Detection Rates

| channel | raw_messages | conflict_events | detection_rate |
| --- | --- | --- | --- |
| KyivIndependent_official | 463 | 153 | 33.0% |
| OSINTdefender | 334 | 70 | 21.0% |
| MiddleEastEye_TG | 268 | 69 | 25.7% |
| liveuamap | 152 | 54 | 35.5% |
| aljazeeraenglish | 50 | 5 | 10.0% |
| bellingcat_en | 50 | 7 | 14.0% |
| demo_security | 1 | 0 | 0.0% |

## Quality Flags

| flag | count | rate_among_conflict_events |
| --- | --- | --- |
| missing_country | 28 | 7.8% |
| missing_location | 28 | 7.8% |
| missing_coordinates | 28 | 7.8% |
| country_centroid_fallback | 178 | 49.7% |
| low_confidence_lt_0.75 | 12 | 3.4% |
| high_severity_low_confidence | 10 | 2.8% |
| negation_or_uncertainty_suspects | 4 | 1.1% |
| older_than_30_days | 146 | 40.8% |
| possible_duplicate_groups | 6 | - |

## Top Matched Keywords

| keyword | count |
| --- | --- |
| attack | 124 |
| drone | 110 |
| killed | 86 |
| missile | 81 |
| wounded | 26 |
| airstrike | 11 |
| protest | 10 |
| casualties | 10 |
| troop | 8 |
| battle | 7 |
| rocket | 7 |
| explosion | 7 |
| evacuation | 5 |
| shelling | 4 |
| clash | 2 |

## Samples To Review

### Missing Country

- `KyivIndependent_official` `2026-07-10T02:59:42+00:00` NO_COUNTRY / strike conf=0.77, sev=0.75: ⚡️Key Russian oil refinery halts production following Ukrainian attack, Reuters reports. A key Russian oil refinery in the city of Saratov halted processing on July 8 following... https://t.me/KyivIndependent_official/54168
- `OSINTdefender` `2026-07-08T18:00:09+00:00` NO_COUNTRY / strike conf=0.72, sev=0.75: #USA #Kazakhstan The Yasa Polaris, a Chevron tanker, was indeed struck by a drone near the Black Sea, but Chevron confirmed that oil exports from Kazakhstan to the Black Sea wer... https://t.me/OSINTdefender/19422
- `OSINTdefender` `2026-07-06T10:52:49+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China's test-fire of a long-range missile from a nuclear-powered submarine into the Pacific Ocean occurred on July 6, 2026, and has drawn significant criticism from regio... https://t.me/OSINTdefender/19392
- `OSINTdefender` `2026-07-03T08:52:37+00:00` NO_COUNTRY / military_movement conf=0.72, sev=0.45: #EU #USA Marco Rubio reportedly helped block Defense Secretary Pete Hegseth's plan to announce significant troop cuts in Europe during a NATO meeting last month. Subscribe to @O... https://t.me/OSINTdefender/19373
- `MiddleEastEye_TG` `2026-07-03T03:04:03+00:00` NO_COUNTRY / conflict_signal conf=0.76, sev=0.75: An Israeli bill to restrict the adhan, the Muslim call to prayer, passed a preliminary reading in parliament on Wednesday. Palestinians condemned the legislation as a "declarati... https://t.me/MiddleEastEye_TG/22113
- `KyivIndependent_official` `2026-07-02T14:05:14+00:00` NO_COUNTRY / conflict_signal conf=0.77, sev=0.75: ⚡️ Around 800,000 of Ukrainian publisher’s books destroyed in overnight Russian attack. According to BookChef, the central warehouse of their logistics partner — where their boo... https://t.me/KyivIndependent_official/54035
- `KyivIndependent_official` `2026-07-02T02:23:27+00:00` NO_COUNTRY / strike conf=0.77, sev=0.83: ⚡️Update: 2 people have been killed and at least 20 others injured in the attack, while drone threats in the city remain ongoing. As of 5 a.m. local time, officials had also rec... https://t.me/KyivIndependent_official/54024
- `OSINTdefender` `2026-06-30T22:26:58+00:00` NO_COUNTRY / strike conf=0.72, sev=0.85: #China The People's Liberation Army Rocket Force (PLARF), formerly the Second Artillery Corps, was established on 1 July 1966, making it nearly 60 years old as of 2026. Subscrib... https://t.me/OSINTdefender/19362

### Missing Coordinates

- `KyivIndependent_official` `2026-07-10T02:59:42+00:00` NO_COUNTRY / strike conf=0.77, sev=0.75: ⚡️Key Russian oil refinery halts production following Ukrainian attack, Reuters reports. A key Russian oil refinery in the city of Saratov halted processing on July 8 following... https://t.me/KyivIndependent_official/54168
- `OSINTdefender` `2026-07-08T18:00:09+00:00` NO_COUNTRY / strike conf=0.72, sev=0.75: #USA #Kazakhstan The Yasa Polaris, a Chevron tanker, was indeed struck by a drone near the Black Sea, but Chevron confirmed that oil exports from Kazakhstan to the Black Sea wer... https://t.me/OSINTdefender/19422
- `OSINTdefender` `2026-07-06T10:52:49+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China's test-fire of a long-range missile from a nuclear-powered submarine into the Pacific Ocean occurred on July 6, 2026, and has drawn significant criticism from regio... https://t.me/OSINTdefender/19392
- `OSINTdefender` `2026-07-03T08:52:37+00:00` NO_COUNTRY / military_movement conf=0.72, sev=0.45: #EU #USA Marco Rubio reportedly helped block Defense Secretary Pete Hegseth's plan to announce significant troop cuts in Europe during a NATO meeting last month. Subscribe to @O... https://t.me/OSINTdefender/19373
- `MiddleEastEye_TG` `2026-07-03T03:04:03+00:00` NO_COUNTRY / conflict_signal conf=0.76, sev=0.75: An Israeli bill to restrict the adhan, the Muslim call to prayer, passed a preliminary reading in parliament on Wednesday. Palestinians condemned the legislation as a "declarati... https://t.me/MiddleEastEye_TG/22113
- `KyivIndependent_official` `2026-07-02T14:05:14+00:00` NO_COUNTRY / conflict_signal conf=0.77, sev=0.75: ⚡️ Around 800,000 of Ukrainian publisher’s books destroyed in overnight Russian attack. According to BookChef, the central warehouse of their logistics partner — where their boo... https://t.me/KyivIndependent_official/54035
- `KyivIndependent_official` `2026-07-02T02:23:27+00:00` NO_COUNTRY / strike conf=0.77, sev=0.83: ⚡️Update: 2 people have been killed and at least 20 others injured in the attack, while drone threats in the city remain ongoing. As of 5 a.m. local time, officials had also rec... https://t.me/KyivIndependent_official/54024
- `OSINTdefender` `2026-06-30T22:26:58+00:00` NO_COUNTRY / strike conf=0.72, sev=0.85: #China The People's Liberation Army Rocket Force (PLARF), formerly the Second Artillery Corps, was established on 1 July 1966, making it nearly 60 years old as of 2026. Subscrib... https://t.me/OSINTdefender/19362

### Low Confidence

- `OSINTdefender` `2026-07-08T18:00:09+00:00` NO_COUNTRY / strike conf=0.72, sev=0.75: #USA #Kazakhstan The Yasa Polaris, a Chevron tanker, was indeed struck by a drone near the Black Sea, but Chevron confirmed that oil exports from Kazakhstan to the Black Sea wer... https://t.me/OSINTdefender/19422
- `OSINTdefender` `2026-07-06T10:52:49+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China's test-fire of a long-range missile from a nuclear-powered submarine into the Pacific Ocean occurred on July 6, 2026, and has drawn significant criticism from regio... https://t.me/OSINTdefender/19392
- `OSINTdefender` `2026-07-03T08:52:37+00:00` NO_COUNTRY / military_movement conf=0.72, sev=0.45: #EU #USA Marco Rubio reportedly helped block Defense Secretary Pete Hegseth's plan to announce significant troop cuts in Europe during a NATO meeting last month. Subscribe to @O... https://t.me/OSINTdefender/19373
- `OSINTdefender` `2026-06-30T22:26:58+00:00` NO_COUNTRY / strike conf=0.72, sev=0.85: #China The People's Liberation Army Rocket Force (PLARF), formerly the Second Artillery Corps, was established on 1 July 1966, making it nearly 60 years old as of 2026. Subscrib... https://t.me/OSINTdefender/19362
- `OSINTdefender` `2026-06-30T18:15:01+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #USA Northrop Grumman has revealed photos of the LGM-35A Sentinel, a new-generation intercontinental ballistic missile designed to replace the aging Minuteman III. This missile... https://t.me/OSINTdefender/19360
- `OSINTdefender` `2026-06-26T08:12:47+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #NK North Korea tested upgraded artillery and missile systems under Kim Jong Un, including an upgraded 240mm multiple rocket launcher with a 90-kilometer range and a tactical ba... https://t.me/OSINTdefender/19330
- `OSINTdefender` `2026-06-21T19:47:33+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #China China has constructed full-scale replicas of the USS Gerald R. Ford aircraft carrier and an Arleigh Burke-class destroyer in the Taklamakan Desert for military training p... https://t.me/OSINTdefender/19296
- `OSINTdefender` `2026-06-17T04:18:54+00:00` NO_COUNTRY / strike conf=0.72, sev=0.90: #USA Navigational warnings indicate a missile test in the Pacific, south of Kwajalein, scheduled for June 27-29, 2026, with a range of approximately 3,800 km. This test is likel... https://t.me/OSINTdefender/19248

### Negation Or Uncertainty Suspects

- `liveuamap` `2026-07-07T13:30:07+00:00` IRN / conflict_signal conf=0.80, sev=0.33: UKMTO has received a report of an incident involving a tanker transiting the Strait of Hormuz. The tanker was struck by an unidentified projectile and is believed to have struct... https://t.me/liveuamap/12211
- `OSINTdefender` `2026-06-18T17:20:41+00:00` UKR / strike conf=0.92, sev=0.75: #Russia #Ukraine #Belarus Will the incident with the bus in the Bryansk region be used for a new round of escalation? Russian and Belarusian officials officially blamed Ukrainia... https://t.me/OSINTdefender/19262
- `OSINTdefender` `2026-06-18T04:31:32+00:00` UKR / conflict_signal conf=0.77, sev=0.37: #Russia #Ukraine Dozens of Ukrainian drones attacked the Moscow Refinery, causing damage but no casualties, according to Mayor Sergey Sobyanin. Emergency services are currently... https://t.me/OSINTdefender/19257
- `KyivIndependent_official` `2026-05-30T19:56:02+00:00` UKR / strike conf=0.97, sev=0.75: ⚡️Kyiv denies Russia's claims of Ukrainian drone strike on Zaporizhzhia nuclear plant. "The version promoted by Russia does not withstand any verification of the facts," the Ukr... https://t.me/KyivIndependent_official/53416

### Older Than 30 Days

- `MiddleEastEye_TG` `2026-06-10T19:05:22+00:00` PSE / conflict_signal conf=0.96, sev=0.75: 📰 Israeli settlers attacked a Palestinian Christian village on Tuesday as their violence across the occupied West Bank continues to escalate. Settlers targeted areas near the vi... https://t.me/MiddleEastEye_TG/21915
- `MiddleEastEye_TG` `2026-06-10T14:02:33+00:00` ISR / conflict_signal conf=0.96, sev=0.75: 📹 “The Israelis were intent on leaving no survivor.” Republican Congressman Thomas Massie has used the 59th anniversary of Israel’s attack on the USS Liberty to call for a new i... https://t.me/MiddleEastEye_TG/21913
- `liveuamap` `2026-06-09T16:50:31+00:00` IRN / conflict_signal conf=0.95, sev=0.75: Following the downing of a U.S. Army AH-64 Apache helicopter near the Strait of Hormuz, U.S. President Donald J. Trump has said that the Apache was in fact shot down while patro... https://t.me/liveuamap/12149
- `liveuamap` `2026-06-09T11:55:09+00:00` LBN / conflict_signal conf=0.95, sev=0.60: Lebanese Ministry of Health: 8 dead and 32 wounded as a result of the Israeli raid that targeted the residential neighborhood in the city of Tyre https://lebanon.liveuamap.com/e... https://t.me/liveuamap/12148
- `liveuamap` `2026-06-09T11:52:34+00:00` IRN / strike conf=0.95, sev=0.75: A U.S. Navy surface drone found and rescued the two crew from a U.S. Apache attack helicopter that crashed into waters near Oman’s coast, the U.S. military’s Central Command tol... https://t.me/liveuamap/12147
- `liveuamap` `2026-06-08T06:53:34+00:00` ISR / strike conf=0.95, sev=0.90: Ballistic missile interceptor seen launching from Central Israel during the recent wave of missiles launched by Iran https://iran.liveuamap.com/en/2026/8-june-06-ballistic-missi... https://t.me/liveuamap/12142
- `liveuamap` `2026-06-08T02:57:19+00:00` ISR / strike conf=0.95, sev=0.90: Israeli army: A missile was launched from Yemen towards Israel, and defense systems are working to intercept it. https://iran.liveuamap.com/en/2026/8-june-02-israeli-army-a-miss... https://t.me/liveuamap/12141
- `liveuamap` `2026-06-07T13:15:29+00:00` LBN / strike conf=0.95, sev=0.85: The Israeli Air Force carried out strikes in Beirut's southern suburbs a short while ago. Prime Minister Benjamin Netanyahu and Defense Minister Israel Katz in a statement say t... https://t.me/liveuamap/12139

### Possible Duplicate Groups

- Group 1: 4 events
  - `KyivIndependent_official` `2026-07-07T22:32:22+00:00` UKR / strike conf=0.97, sev=0.75: ⚡️Raid on leading drone company fuels fears of crackdown on Ukraine's defense tech sector. The CEO of top Ukrainian drone firm Vyriy Industries denied allegations of price gougi... https://t.me/KyivIndependent_official/54122
  - `KyivIndependent_official` `2026-07-07T17:39:31+00:00` UKR / strike conf=0.97, sev=0.75: ⚡️Update: Ukraine signs drone cooperation deal with Denmark. President Volodymyr Zelensky and Danish Prime Minister Mette Frederiksen also discussed strengthening air defenses a... https://t.me/KyivIndependent_official/54116
  - `OSINTdefender` `2026-07-07T17:16:37+00:00` UKR / strike conf=0.92, sev=0.75: #Ukraine #Russia #EU Ukraine's Unmanned Systems Forces have conducted drone strikes on tankers in the Azov Sea, bringing the total number of tankers struck today to 12. This esc... https://t.me/OSINTdefender/19407
  - `KyivIndependent_official` `2026-07-07T16:31:14+00:00` UKR / strike conf=0.97, sev=0.75: ⚡️Ukraine signs drone cooperation deals with Estonia, Netherlands. The agreements are part of Ukraine's broader push to expand drone technology cooperation with partners in Euro... https://t.me/KyivIndependent_official/54114
- Group 2: 3 events
  - `OSINTdefender` `2026-07-03T17:45:11+00:00` UKR / conflict_signal conf=0.92, sev=0.75: #Ukraine #USA Zelensky emphasizes the need for Ukraine to establish its own production of Patriot missiles to enhance air defense capabilities following a significant Russian at... https://t.me/OSINTdefender/19379
  - `KyivIndependent_official` `2026-07-03T13:10:56+00:00` UKR / conflict_signal conf=0.97, sev=0.75: ⚡️Kyiv air quality plunges after Russia's biggest attack on capital of entire war. The smell of smoke and haze lingered over the capital. Authorities urge residents to keep wind... https://t.me/KyivIndependent_official/54053
  - `KyivIndependent_official` `2026-07-03T12:03:45+00:00` UKR / conflict_signal conf=0.97, sev=0.75: ⚡️Update: Monaco issues arrest warrant for Ukrainian woman suspected of bomb attack that injured Ukraine-born businessman. Monaco issued an Interpol "red notice" — an internatio... https://t.me/KyivIndependent_official/54052
- Group 3: 2 events
  - `OSINTdefender` `2026-06-28T05:22:35+00:00` USA / conflict_signal conf=0.92, sev=0.75: #Kuwait #Bahrain Iranian ballistic missiles and drones reportedly struck 8 US military targets, including Ali Al Salem Air Base in Kuwait and the US Navy's 5th Fleet base in Bah... https://t.me/OSINTdefender/19344
  - `liveuamap` `2026-06-28T02:56:07+00:00` USA / conflict_signal conf=0.95, sev=0.75: In a statement, Iran's Islamic Revolutionary Guard Corps (IRGC) has confirmed that it attacked US forces in two military bases in the Middle East, the Ali Al Salem air base in K... https://t.me/liveuamap/12192
- Group 4: 2 events
  - `OSINTdefender` `2026-06-22T18:09:00+00:00` UKR / conflict_signal conf=0.92, sev=0.75: #Russia #Ukraine On June 22, 2026, Ukrainian Storm Shadow cruise missiles struck the Voronezh Semiconductor Devices Plant, causing extensive damage, a major fire, and reportedly... https://t.me/OSINTdefender/19305
  - `KyivIndependent_official` `2026-06-22T12:33:09+00:00` UKR / conflict_signal conf=0.97, sev=0.75: ⚡️ Russian attack damages production facility of Ukraine’s FPV giant. “This is war. We were prepared for such events,” the company’s founder, Yaroslav Gryshyn, said. “The enemy... https://t.me/KyivIndependent_official/53828
- Group 5: 2 events
  - `OSINTdefender` `2026-05-29T17:55:26+00:00` ROU / strike conf=0.92, sev=0.75: #Germany #Romania #Russia German Chancellor Friedrich Merz emphasized that Germany is prepared to defend the territory of NATO allies, particularly in light of recent security c... https://t.me/OSINTdefender/19056
  - `KyivIndependent_official` `2026-05-29T12:58:59+00:00` ROU / strike conf=0.97, sev=0.75: ⚡️NATO condemns ‘reckless’ Russian drone strike on Romanian residential building NATO Secretary General Mark Rutte has said "Russia's reckless behaviour is a danger to us all,"... https://t.me/KyivIndependent_official/53396
- Group 6: 2 events
  - `OSINTdefender` `2026-05-29T05:08:37+00:00` ROU / strike conf=0.92, sev=0.75: #Romania A Russian drone crashed in Galați, Romania, causing minor damage to a residential building and injuring two people. The drone was reported to have an explosive payload,... https://t.me/OSINTdefender/19045
  - `KyivIndependent_official` `2026-05-29T01:27:35+00:00` ROU / strike conf=0.97, sev=0.75: ⚡️Drone reportedly strikes residential building in Romania. A drone reportedly struck a residential building in Galati, Romania, overnight on May 29, news outlet Viata Libera re... https://t.me/KyivIndependent_official/53391

## Suggested Next Actions

- If missing coordinates are high, expand `CITY_COORDS` and country aliases in `src/live_osint/extraction.py`.
- If old events are high, add a time-window filter during export or collection.
- If low-confidence events are useful, lower display threshold; if noisy, raise it.
- If one keyword dominates false positives, reduce its weight or require a country/location match.
