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
- why: 같은 레시피에서 seed만 바꿨을 때 결과가 얼마나 흔들리는지를 재서 앞으로 볼 정확도 개선이 진짜인지 운인지 판단할 기준을 만들려는 것이다.
- validation: seed 0 0.4894, seed 1 0.4958 (seoul 0.573/0.570, busan 0.375/0.387, tokyo 0.637/0.614, osaka 0.330/0.381, taipei 0.531/0.527)
- what it means: seed만 바꿔서 두 번 돌려 봤더니 전체 점수가 0.489, 0.496으로 거의 똑같이 나왔다. 따라서 기본 성능은 0.49 정도라고 보면 될 것 같다. 도시별로 보면 점수 변동이 좀 더 큰데, 오사카는 seed만 바꿨는데 5%p 차이가 났다. 높은 점수 순서는 학습 사진 수랑 거의 비슷했다. 사진이 제일 많은 도쿄와 서울이 점수도 제일 높았고, 제일 적은 부산은 낮았다. 이때 오사카와 타이베이는 사진 수가 거의 같은데도 점수가 0.330~0.381, 0.527~0.531로 차이가 많이 났다. 그 이유는 학습 사진이 많은 도쿄로 인해 오사카가 밀린 것으로 유추해볼 수 있다. 모델이 어느 나라인지는 대강 맞히지만, 헷갈리는 사진은 같은 나라의 학습 사진이 많은 도시 쪽으로 분류하는 것이다. 이 때문에 오사카는 도쿄에게, 부산은 서울에게 밀리고, 같은 나라에 경쟁 도시가 없는 타이베이는 그러한 손해를 보지 않는 것 같다.

## 2026-11-03 What tells the cities apart (my look at 100 training photos per city)

서울
고층건물, 회색, 흐린 하늘, 한글 간판, 한옥, 거대한 문들(ex 장안문), 빽빽한 건물들, 한국 지하철, 건물 뒤편 산, 평지에 이어진 아파트 단지

부산
바다, 한글 간판, 파란 버스, 부산 시장, 건물 뒤편 산, 바다와 산비탈에 층층이 들어선 주택, 한옥

도쿄
바다, 일본 간판, 좌측통행 차도, 일본식 지하철들, 촘촘한 전봇대와 전선

오사카
일본 간판, 도톤보리 강, 밤에 빛나는 간판 거리, 좌측통행 차도, 낮은 건물, 일본식 지하철들, 촘촘한 전봇대와 전선

타이베이
낮은 건물, 야시장, 전통 가옥, 오토바이와 스쿠터 개많음, 우림, 울창한 아열대 나무, 넓은 차도, 건물 뒤편 산
