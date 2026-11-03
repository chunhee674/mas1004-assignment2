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

## 2026-11-03 What the baseline's mistakes have in common (mistakes_baseline_seed0, all five sheets)

1. 도시를 구분할 단서가 없는 지극히 평범한 사진, 촬영 조건이 특이한 사진을 틀린다. 차나 음식을 가까이 찍어서 물체만 보이는 경우, 어느 나라에나 있을 법한 도로나 숲길, 배경이 보이지 않는 실내의 경우가 주로 확인되며 이 경우에는 사람이 봐도 어느 도시인지 알기 어렵다.
예: 서울 #3(차 클로즈업), 서울 #11(음식 클로즈업), 서울 #14(숲길), 타이베이 #16(실내 로비) 또한 촬영 조건이 특이한 사진은 장소와 상관없이 찍는 방식이나 날씨 때문에 화면 전체의 색감과 형태가 크게 달라진 사진이다. 차 안에서 앞유리 너머로 찍은 블랙박스 화면, 둥글게 휘어진 360도 파노라마, 빗물에 번지고 어두운 비 오는 날, 불빛 말고는 거의 안 보이는 밤 사진 등이 여기에 해당한다. 모델은 건물이나 거리 모습보다 이런 외부적 조건을 중심으로 도시를 고르는 것으로 보인다.
예: 타이베이 #2(블랙박스) -> 도쿄 0.98, 타이베이 #12(360도 파노라마) -> 서울 0.88, 도쿄 #10(비) -> 서울 0.92, 서울 #19(밤 주행) -> 타이베이 0.82

2. 간판이 있어 어느 나라 글자인지 보이는데도 다른 나라로 틀린다. 간판이나 표지판에 한글, 일본어(가나), 번체 한자 중 하나가 분명히 보이는 사진이기에 어떤 문자인지만 구분하면 나라를 맞힐 수 있는데, 모델은 그 단서를 쓰지 못하고 엉뚱한 나라를 골랐다.
예: 부산 #2(한글 간판 시장), 오사카 #4(일본어 표지판)

## 2026-11-03 balanced (Track A)
- changed: every city drawn equally often (`sample_weights` gives each photo 1 / the number of photos of its city)
- why: (to ask the student)
- validation: 0.5139 (before: baseline 0.4894 seed 0, 0.4958 seed 1); seoul 0.517, busan 0.467, tokyo 0.529, osaka 0.444, taipei 0.614
- what it means: (to ask the student)
