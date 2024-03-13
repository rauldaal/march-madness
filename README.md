# march-madness

## Data Explanation

### 2024_tourney_seeds.csv
|Columns| Desctiption |
|-|-|
|Tourment  | Si es Masculino o Femenino|
|Seed| Region + Seed number (Ejemplo: W01)  |
|TeamId| Identificador Team |


### MConferencesTourneyGames.csv
|Columns| Desctiption |
|-|-|
|Season  | Año|
|ConfAbbrev| Abreviatura conferencia  |
|DayNum| Numero de partido (120 ==> dt(KickOffDay)+120) |
|WTeamID| ID equipo ganador|
|LTeamID| ID equipo perdedor|


### MGamesCities.csv
|Columns| Desctiption |
|-|-|
|Season  | Año|
|DayNum  | Numero de partido|
|WTeamID| ID equipo ganador|
|LTeamID| ID equipo perdedor|
|CRType| Tipo de partido (NCAA, Regular, Secondary)|
|CityID| ID Ciudad|


### MMasseyOrdinals.csv
|Columns| Desctiption |
|-|-|
|Season  | Año|
|RankingDayNum  | Tiene en cuenta el los n-1 partidos anteriores|
|SystemName| Sistema de ranking|
|TeamId| ID equipo|
|OrdinalRank| Posición en el Ranking|

### MNCAATourneyCompactResults.csv
|Columns| Desctiption |
|-|-|
|Season  | Año|
|DayNum  | Numero de partido|
|WTeamID| ID equipo ganador|
|WScore| Marcador equipo ganador|
|LTeamID| ID equipo perdedor|
|LScore| Marcador equipo perdedor|
|WLoc| Home (H), Away (A), Neutral (N) respecto al ganador|
|NumOT| Numero Prorrogas|

### MNCAATourneyDetailedResults.csv
|Columns| Desctiption |
|-|-|
|Season  | Año|
|DayNum  | Numero de partido|
|WTeamID| ID equipo ganador|
|WScore| Marcador equipo ganador|
|LTeamID| ID equipo perdedor|
|LScore| Marcador equipo perdedor|
|WLoc| Home (H), Away (A), Neutral (N) respecto al ganador|
|NumOT| Numero Prorrogas|
|FGM|Tiros de campo anotados|
|FGA|Tiros de campo intentos|
|FGM3|Tiros de campo anotados 3pts|
|FGA3|Tiros de campo intentos 3pts|
|FTM|Tiros libres anotados|
|FTA|Tiros libres intentados|
|OR|Rebotes ofensivos|
|DR|Rebotes defensivos|
|Ast|Asistencias|
|TO|Turn Overs (perdidas de balon)|
|Stl|Robos|
|Blk|Bloqueos|
|PF|Faltas personales|

### MNCAATourneySeedRoundSlots.csv

|Columns| Desctiption |
|-|-|
|Seed|Region + Seed number (Ejemplo: W01) |
|Game Round| Momento en el torneo cuando se jugaría|
|GameSlot| Slot del partido|
|EasrlyDay| Numero del dia como muy temprano|
|LateDay| Numero del dia como muy tarde|


### MNCAATourneySeeds.csv
|Columns| Desctiption |
|-|-|
|Season  | Año|
|Seed  | Seed del equipo|
|TeamId  | Identificador equipo|

### MNCAATourneySlots.csv
|Columns| Desctiption |
|-|-|
|Season  | Año|
|Slor  | Slot del partido|
|StrongSeed  | Seed del equipo favorito|
|WeakSeed| Seed del equipo underdog


### MTeamCoaches.csv
|Columns| Desctiption |
|-|-|
|Season| Año|
|TeamID| ID Equipo|
|FirstDayNum|Matchday primer partido|
|LastDayNum| Matchday ultimo partido|
|CoachName|Nombre entrenador|

### MTeamConferences.csv
|Columns| Desctiption |
|-|-|
|Season| Año|
|TeamID| Identificador equipo|
|ConfAbbrev| Abreviatura Conferencia|

### MTeams.csv
|Columns| Desctiption |
|-|-|
|TeamID|Identificador equipo|
|TeamName|Nombre de equipo|
|FirstD1Season|Primera vez en division1|
|LastD1Season|Ultima vez en division 1|