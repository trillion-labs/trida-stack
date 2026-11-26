"""Topic cards for Zoro Draw, in decks the host picks in the lobby.

words      English words with hand-picked lookalike decoys (a quick sketch of the word could be mistaken for
           them; random decoys make every drawing score ~1.0). 6 decoys each: classic uses all 6, decoy mode 3.
proverbs   한국어 속담. Draw the literal picture.
ko_memes   한국 밈 · 신조어.
memes      Internet memes.
For the last three, decoys are other cards from the same deck, and each card has an English gloss describing the
picture. Zoro sees "card: gloss" as the option text (its Korean is weak; the gloss tells it what to look for);
players see the card, and the drawer also sees the gloss.
"""
import random
CARDS: dict[str, list[str]] = {
    "lighthouse": ["tower", "rocket", "candle", "chimney", "castle", "lamp post"],
    "snowman": ["cloud", "bear", "ghost", "bowling pin", "scarecrow", "marshmallow"],
    "jellyfish": ["octopus", "umbrella", "mushroom", "squid", "ghost", "lamp"],
    "giraffe": ["horse", "deer", "camel", "dinosaur", "llama", "zebra"],
    "volcano": ["mountain", "pyramid", "tent", "fountain", "campfire", "anthill"],
    "guitar": ["violin", "ukulele", "banjo", "cello", "tennis racket", "frying pan"],
    "snail": ["shell", "slug", "turtle", "cinnamon roll", "ammonite", "hedgehog"],
    "windmill": ["fan", "pinwheel", "helicopter", "flower", "propeller", "clock tower"],
    "penguin": ["duck", "owl", "bowling pin", "seal", "crow", "puffin"],
    "cactus": ["tree", "pickle", "candelabra", "coral", "broccoli", "hand"],
    "headphones": ["earmuffs", "horseshoe", "magnet", "rainbow", "stethoscope", "glasses"],
    "submarine": ["whale", "blimp", "fish", "rocket", "bus", "torpedo"],
    "octopus": ["spider", "jellyfish", "squid", "sun", "tree roots", "starfish"],
    "rainbow": ["bridge", "arch", "headphones", "smile", "horseshoe", "tunnel"],
    "hot air balloon": ["light bulb", "ice cream cone", "parachute", "lollipop", "acorn", "kite"],
    "kangaroo": ["rabbit", "dinosaur", "deer", "mouse", "dog", "squirrel"],
    "anchor": ["arrow", "trident", "umbrella", "sword", "hook", "cross"],
    "mushroom": ["umbrella", "tree", "jellyfish", "lamp", "table", "ice cream"],
    "crown": ["castle", "fence", "comb", "flame", "saw", "mountains"],
    "skateboard": ["surfboard", "car", "train", "sled", "bench", "pencil case"],
    "spider": ["octopus", "ant", "crab", "sun", "bug", "star"],
    "pineapple": ["grenade", "pine cone", "palm tree", "carrot", "hedgehog", "artichoke"],
    "ladder": ["fence", "railway", "zipper", "spine", "bridge", "bookshelf"],
    "tornado": ["funnel", "ice cream cone", "spring", "drill", "smoke", "vase"],
    "dragon": ["dinosaur", "lizard", "bat", "crocodile", "bird", "snake"],
    "toothbrush": ["comb", "brush", "spoon", "razor", "pen", "broom"],
    "igloo": ["dome", "turtle", "bowl", "helmet", "tent", "bread"],
    "scissors": ["glasses", "pliers", "bicycle", "crab", "tongs", "airplane"],
    "owl": ["cat", "penguin", "bat", "bird", "bear", "egg"],
    "teapot": ["kettle", "watering can", "vase", "lamp", "mug", "bird"],
    "bicycle": ["glasses", "motorcycle", "wheelchair", "scooter", "car", "dumbbell"],
    "palm tree": ["coconut", "firework", "cactus", "pineapple", "umbrella", "fountain"],
    "shark": ["dolphin", "fish", "whale", "submarine", "boat", "airplane"],
    "candle": ["lighthouse", "torch", "matchstick", "lipstick", "rocket", "bottle"],
    "hedgehog": ["porcupine", "cactus", "mouse", "pineapple", "brush", "sun"],
    "bee": ["fly", "wasp", "ladybug", "butterfly", "tiger", "bird"],
    "castle": ["fortress", "crown", "church", "tower", "chess rook", "sandcastle"],
    "umbrella": ["mushroom", "jellyfish", "anchor", "parachute", "tent", "lamp"],
    "astronaut": ["robot", "diver", "snowman", "teddy bear", "knight", "beekeeper"],
    "pizza": ["pie", "cheese", "clock", "sun", "watermelon slice", "flag"],
}


PROVERBS: dict[str, str] = {
    "고래 싸움에 새우 등 터진다": "two whales fighting while a small shrimp between them gets crushed",
    "원숭이도 나무에서 떨어진다": "a monkey falling out of a tree",
    "낮말은 새가 듣고 밤말은 쥐가 듣는다": "a bird listening by day and a mouse listening by night",
    "소 잃고 외양간 고친다": "fixing an empty barn after the cow has run away",
    "가는 말이 고와야 오는 말이 곱다": "kind words going out and kind words coming back",
    "하늘의 별 따기": "someone trying to pick a star out of the sky",
    "개구리 올챙이 적 생각 못 한다": "a frog that forgot it used to be a tadpole",
    "고양이 목에 방울 달기": "mice trying to hang a bell on a cat's neck",
    "우물 안 개구리": "a frog sitting at the bottom of a well",
    "그림의 떡": "a rice cake inside a painting that you can't eat",
    "누워서 떡 먹기": "someone lying down eating rice cake",
    "식은 죽 먹기": "easily eating a bowl of cold porridge",
    "빈 수레가 요란하다": "an empty cart making a lot of noise",
    "등잔 밑이 어둡다": "darkness right under an oil lamp",
    "호랑이도 제 말 하면 온다": "people talking about a tiger as the tiger shows up",
    "꿩 먹고 알 먹고": "eating a pheasant and its eggs at the same time",
    "떡 줄 사람은 생각도 않는데 김칫국부터 마신다": "drinking kimchi soup while waiting for rice cake nobody offered",
    "바늘 도둑이 소도둑 된다": "a thief who steals a needle grows up to steal a cow",
    "세 살 버릇 여든까지 간다": "a 3-year-old and an 80-year-old with the same habit",
    "서당 개 삼 년이면 풍월을 읊는다": "a dog at a village school reciting poetry",
    "발 없는 말이 천 리 간다": "words (or a horse) without feet traveling very far",
    "콩 심은 데 콩 나고 팥 심은 데 팥 난다": "planting beans and getting beans, planting red beans and getting red beans",
    "닭 쫓던 개 지붕 쳐다본다": "a dog staring up at a chicken on the roof",
    "소 귀에 경 읽기": "someone reading scripture into a cow's ear",
    "백지장도 맞들면 낫다": "two people lifting a single sheet of paper together",
    "금강산도 식후경": "eating first before enjoying the mountain view",
    "사공이 많으면 배가 산으로 간다": "a boat with too many rowers going up a mountain",
    "티끌 모아 태산": "collecting specks of dust into a huge mountain",
    "돌다리도 두들겨 보고 건너라": "tapping a stone bridge before crossing it",
    "꼬리가 길면 밟힌다": "a very long tail getting stepped on",
    "벼는 익을수록 고개를 숙인다": "ripe rice plants bowing their heads",
}

KO_MEMES: dict[str, str] = {
    "얼죽아": "drinking iced americano even while freezing in winter",
    "오운완": "a gym selfie after finishing today's workout",
    "중꺾마": "a heart that won't break (never give up)",
    "갑분싸": "the mood in the room suddenly turning cold and awkward",
    "킹받네": "someone extremely annoyed, wearing a crown of rage",
    "어쩔티비": "someone shrugging 'so what?' and pointing at a TV",
    "존버": "holding on to something no matter what",
    "꾸안꾸": "someone carefully dressed up to look effortless",
    "점메추": "asking for a lunch menu recommendation",
    "퇴근각": "staring at the clock, ready to leave work",
    "혼밥": "eating alone at a restaurant",
    "소확행": "a small but certain happiness, like a warm cup of coffee",
    "워라밸": "a scale balancing work and life",
    "영끌": "scraping together every last coin to buy an apartment",
    "월급루팡": "a sneaky employee stealing a paycheck while doing nothing",
    "치맥": "fried chicken and beer",
    "불금": "a Friday night on fire",
    "금사빠": "someone who falls in love instantly with hearts in their eyes",
    "깻잎 논쟁": "someone helping a friend's partner separate perilla leaves",
    "민초단": "a team of mint chocolate lovers",
    "현타": "a sudden reality check, staring into space",
    "오픈런": "people running into a store the moment it opens",
    "랜선집사": "someone loving a cat through a phone screen",
}

MEMES: dict[str, str] = {
    "this is fine": "a dog sipping coffee in a room on fire",
    "distracted boyfriend": "a man turning to look at another woman while his girlfriend glares at him",
    "drake yes/no": "a man rejecting one thing with his hand and approving another",
    "woman yelling at a cat": "a woman yelling and a confused cat at a dinner table",
    "expanding brain": "a brain glowing brighter and brighter",
    "two buttons": "a sweating man choosing between two red buttons",
    "is this a pigeon?": "a man pointing at a butterfly",
    "stonks": "a man in a suit in front of a rising stock chart",
    "doge": "a shiba inu dog with colorful text around it",
    "grumpy cat": "a frowning cat",
    "rickroll": "a man in a trench coat singing and dancing",
    "nyan cat": "a cat with a toaster-pastry body flying with a rainbow trail",
    "success kid": "a toddler on a beach clenching a fist",
    "disaster girl": "a girl smiling in front of a burning house",
    "hide the pain harold": "an old man smiling while clearly in pain",
    "change my mind": "a man sitting at a table with a sign",
    "trollface": "a wide mischievous grinning face",
    "keyboard cat": "a cat playing a keyboard",
    "ice bucket challenge": "pouring a bucket of ice water over your head",
    "planking": "a person lying stiff and face down on a weird object",
    "gangnam style": "a man doing a horse-riding dance",
}

DECKS: dict[str, dict] = {
    "words": {"label": "Words", "cards": {w: None for w in CARDS}},
    "proverbs": {"label": "한국어 속담", "cards": PROVERBS},
    "ko_memes": {"label": "한국 밈 · 신조어", "cards": KO_MEMES},
    "memes": {"label": "Internet memes", "cards": MEMES},
}
DECK_OF = {w: d for d, deck in DECKS.items() for w in deck["cards"]}


def gloss(word: str) -> str | None:
    """English description of a card for Zoro (None for plain words and player-written decoys)."""
    d = DECK_OF.get(word)
    return DECKS[d]["cards"].get(word) if d else None


def decoys(word: str, k: int, rng: random.Random | None = None) -> list[str]:
    """k decoys for a card: hand-picked lookalikes for words, other cards of the same deck otherwise."""
    if word in CARDS:
        return CARDS[word][:k]
    others = [w for w in DECKS[DECK_OF[word]]["cards"] if w != word]
    return (rng or random).sample(others, min(k, len(others)))
