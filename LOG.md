# Log

One entry for each recipe, selection or Track B model you train, newest at
the bottom. The numbers come from `experiments.csv`. "Why" and "what it means"
are in your own words: your agent asks you for them and does not make them up.
End the log with your plan for what to try next.

<!-- The form of an entry:

## <date> <tag> (Track A or B)
- changed: what is different from the run before
- why: your reason
- validation: 0.000 (before: 0.000)
- what it means: what you think the result says
-->

## 2026-11-01 baseline (Track A)
- changed: nothing, the starter recipe on all training photos; seeds 0 and 1
- why: (to ask the student)
- validation: seed 0 0.4894, seed 1 0.4958 (seoul 0.573/0.570, busan 0.375/0.387, tokyo 0.637/0.614, osaka 0.330/0.381, taipei 0.531/0.527)
- what it means: (to ask the student)

## 2026-11-03 What tells the cities apart (my look at 100 training photos per city)

서울
고층건물, 회색, 흐린 하늘, 한글 간판, 한옥, 거대한 문들(ex 장안문), 빽빽한 건물들, 한국 지하철, 건물 뒤편 산, 평지에 이어진 아파트 단지

부산
바다, 한글 간판, 파란 버스, 부산 시장, 건물 뒤편 산, 바다와 산비탈에 층층이 들어선 주택, 한옥

도쿄
바다, 비교적 건물, 일본 간판, 좌측통행 차도, 일본식 지하철들, 촘촘한 전봇대와 전선

오사카
일본 간판, 도톤보리 강, 밤에 빛나는 간판 거리, 좌측통행 차도, 낮은 건물, 일본식 지하철들, 촘촘한 전봇대와 전선

타이베이
낮은 건물, 야시장, 전통 가옥, 오토바이와 스쿠터 개많음, 우림, 울창한 아열대 나무, 넓은 차도, 건물 뒤편 산
