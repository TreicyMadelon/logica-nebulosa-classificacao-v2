"""Tabela de códigos do ImageNet.

No XML do ImageNet, a categoria vem como código (WNID), por exemplo
n02342885 = hamster. Esta tabela traduz os códigos usados no projeto.

As 1.000 categorias do ImageNet seguem a ordem dos códigos:
- 0 a 397 são animais (até n02655020); a 398 (ábaco, n02666196) é o primeiro objeto;
- 151 a 268 são as 118 raças domésticas de cachorro (n02085620 a n02113978).
"""

PRIMEIRA_RACA = "n02085620"     # Chihuahua (categoria 151)
ULTIMA_RACA = "n02113978"       # Mexican hairless (categoria 268)
PRIMEIRO_OBJETO = "n02666196"   # abacus (categoria 398)


def e_raca_de_cachorro(codigo):
    """Raça doméstica de cachorro (rótulo cachorro)."""
    return PRIMEIRA_RACA <= codigo <= ULTIMA_RACA


def e_animal(codigo):
    """Animal do ImageNet (categorias 0 a 397)."""
    return codigo < PRIMEIRO_OBJETO


# Não cachorro · outros animais: o hamster, os canídeos selvagens e outras espécies.
OUTROS_ANIMAIS_IMAGENET = {
    "n02342885": "hamster",
    # canídeos selvagens: são não cachorro (bons candidatos à área cinzenta)
    "n02114367": "lobo-cinzento",
    "n02114548": "lobo-branco",
    "n02114712": "lobo-vermelho",
    "n02114855": "coiote",
    "n02115641": "dingo",
    "n02116738": "mabeco",
    "n02119022": "raposa-vermelha",
    "n02119789": "raposa (kit fox)",
    "n02120079": "raposa-do-ártico",
    "n02120505": "raposa-cinzenta",
    # outras espécies
    "n02364673": "porquinho-da-índia",
    "n02325366": "coelho",
    "n02356798": "esquilo",
    "n02129165": "leão",
    "n02129604": "tigre",
    "n02132136": "urso-pardo",
    "n02504458": "elefante-africano",
    "n02391049": "zebra",
    "n02510455": "panda-gigante",
    "n02481823": "chimpanzé",
    "n02437616": "lhama",
    "n02444819": "lontra",
    "n02138441": "suricato",
    "n01882714": "coala",
}

# Não cachorro · objetos e cenários.
OBJETOS_E_CENARIOS_IMAGENET = {
    "n02690373": "avião",
    "n02701002": "ambulância",
    "n03792782": "bicicleta",
    "n04285008": "carro esportivo",
    "n04335435": "bonde",
    "n04147183": "veleiro",
    "n03063599": "caneca",
    "n03642806": "notebook",
    "n04557648": "garrafa de água",
    "n03028079": "igreja",
    "n04366367": "ponte pênsil",
    "n02708093": "relógio",
    "n03452741": "piano de cauda",
    "n02676566": "violão",
    "n04070727": "geladeira",
    "n04442312": "torradeira",
    "n06874185": "semáforo",
    "n09193705": "montanha",
    "n09472597": "vulcão",
    "n09332890": "beira de lago",
    "n09468604": "vale",
    "n07753592": "banana",
    "n07873807": "pizza",
    "n13054560": "cogumelo",
    "n11939491": "margarida",
}

# Cachorro · as 118 raças domésticas (nomes oficiais do ImageNet, em inglês).
RACAS_DE_CACHORRO = {
    "n02085620": "Chihuahua", "n02085782": "Japanese spaniel", "n02085936": "Maltese dog",
    "n02086079": "Pekinese", "n02086240": "Shih-Tzu", "n02086646": "Blenheim spaniel",
    "n02086910": "papillon", "n02087046": "toy terrier", "n02087394": "Rhodesian ridgeback",
    "n02088094": "Afghan hound", "n02088238": "basset", "n02088364": "beagle",
    "n02088466": "bloodhound", "n02088632": "bluetick", "n02089078": "black-and-tan coonhound",
    "n02089867": "Walker hound", "n02089973": "English foxhound", "n02090379": "redbone",
    "n02090622": "borzoi", "n02090721": "Irish wolfhound", "n02091032": "Italian greyhound",
    "n02091134": "whippet", "n02091244": "Ibizan hound", "n02091467": "Norwegian elkhound",
    "n02091635": "otterhound", "n02091831": "Saluki", "n02092002": "Scottish deerhound",
    "n02092339": "Weimaraner", "n02093256": "Staffordshire bullterrier",
    "n02093428": "American Staffordshire terrier", "n02093647": "Bedlington terrier",
    "n02093754": "Border terrier", "n02093859": "Kerry blue terrier", "n02093991": "Irish terrier",
    "n02094114": "Norfolk terrier", "n02094258": "Norwich terrier", "n02094433": "Yorkshire terrier",
    "n02095314": "wire-haired fox terrier", "n02095570": "Lakeland terrier",
    "n02095889": "Sealyham terrier", "n02096051": "Airedale", "n02096177": "cairn",
    "n02096294": "Australian terrier", "n02096437": "Dandie Dinmont", "n02096585": "Boston bull",
    "n02097047": "miniature schnauzer", "n02097130": "giant schnauzer",
    "n02097209": "standard schnauzer", "n02097298": "Scotch terrier", "n02097474": "Tibetan terrier",
    "n02097658": "silky terrier", "n02098105": "soft-coated wheaten terrier",
    "n02098286": "West Highland white terrier", "n02098413": "Lhasa",
    "n02099267": "flat-coated retriever", "n02099429": "curly-coated retriever",
    "n02099601": "golden retriever", "n02099712": "Labrador retriever",
    "n02099849": "Chesapeake Bay retriever", "n02100236": "German short-haired pointer",
    "n02100583": "vizsla", "n02100735": "English setter", "n02100877": "Irish setter",
    "n02101006": "Gordon setter", "n02101388": "Brittany spaniel", "n02101556": "clumber",
    "n02102040": "English springer", "n02102177": "Welsh springer spaniel",
    "n02102318": "cocker spaniel", "n02102480": "Sussex spaniel", "n02102973": "Irish water spaniel",
    "n02104029": "kuvasz", "n02104365": "schipperke", "n02105056": "groenendael",
    "n02105162": "malinois", "n02105251": "briard", "n02105412": "kelpie", "n02105505": "komondor",
    "n02105641": "Old English sheepdog", "n02105855": "Shetland sheepdog", "n02106030": "collie",
    "n02106166": "Border collie", "n02106382": "Bouvier des Flandres", "n02106550": "Rottweiler",
    "n02106662": "German shepherd", "n02107142": "Doberman", "n02107312": "miniature pinscher",
    "n02107574": "Greater Swiss Mountain dog", "n02107683": "Bernese mountain dog",
    "n02107908": "Appenzeller", "n02108000": "EntleBucher", "n02108089": "boxer",
    "n02108422": "bull mastiff", "n02108551": "Tibetan mastiff", "n02108915": "French bulldog",
    "n02109047": "Great Dane", "n02109525": "Saint Bernard", "n02109961": "Eskimo dog",
    "n02110063": "malamute", "n02110185": "Siberian husky", "n02110341": "dalmatian",
    "n02110627": "affenpinscher", "n02110806": "basenji", "n02110958": "pug",
    "n02111129": "Leonberg", "n02111277": "Newfoundland", "n02111500": "Great Pyrenees",
    "n02111889": "Samoyed", "n02112018": "Pomeranian", "n02112137": "chow", "n02112350": "keeshond",
    "n02112706": "Brabancon griffon", "n02113023": "Pembroke", "n02113186": "Cardigan",
    "n02113624": "toy poodle", "n02113712": "miniature poodle", "n02113799": "standard poodle",
    "n02113978": "Mexican hairless",
}


def nome_imagenet(codigo):
    """Nome legível de um código do ImageNet (ou o próprio código, se não estiver na tabela)."""
    for tabela in (RACAS_DE_CACHORRO, OUTROS_ANIMAIS_IMAGENET, OBJETOS_E_CENARIOS_IMAGENET):
        if codigo in tabela:
            return tabela[codigo]
    return codigo
