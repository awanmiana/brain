"""Seed knowledge for the Pathway Lattice v1.

This is the distilled corpus that gets *compiled* into deterministic rank-1
path operators by `pathway_lattice.py`. It is not training data in the SGD
sense: every triple becomes one exact operator (subject, relation) -> object,
and multi-hop chains emerge because the object of one operator is the subject
of the next.

Domains: geography, elements, science, computing, history, and three designed
reasoning chains (teaches / is-a / knows) that exercise multi-hop path walking.
"""

FACTS: list[tuple[str, str, str]] = [
    # --- geography: capitals (query-natural: "france capital" -> paris) --
    ("france", "capital", "paris"),
    ("italy", "capital", "rome"),
    ("germany", "capital", "berlin"),
    ("japan", "capital", "tokyo"),
    ("spain", "capital", "madrid"),
    ("portugal", "capital", "lisbon"),
    ("norway", "capital", "oslo"),
    ("sweden", "capital", "stockholm"),
    ("finland", "capital", "helsinki"),
    ("greece", "capital", "athens"),
    ("austria", "capital", "vienna"),
    ("switzerland", "capital", "bern"),
    ("czech-republic", "capital", "prague"),
    ("poland", "capital", "warsaw"),
    ("hungary", "capital", "budapest"),
    ("ukraine", "capital", "kyiv"),
    ("russia", "capital", "moscow"),
    ("turkey", "capital", "ankara"),
    ("egypt", "capital", "cairo"),
    ("kenya", "capital", "nairobi"),
    ("canada", "capital", "ottawa"),
    ("mexico", "capital", "mexico-city"),
    ("brazil", "capital", "brasilia"),
    ("argentina", "capital", "buenos-aires"),
    ("australia", "capital", "canberra"),
    ("new-zealand", "capital", "wellington"),
    ("china", "capital", "beijing"),
    ("south-korea", "capital", "seoul"),
    ("india", "capital", "new-delhi"),
    ("pakistan", "capital", "islamabad"),
    ("thailand", "capital", "bangkok"),
    ("vietnam", "capital", "hanoi"),
    ("indonesia", "capital", "jakarta"),
    ("philippines", "capital", "manila"),
    ("ireland", "capital", "dublin"),
    ("united-kingdom", "capital", "london"),
    ("netherlands", "capital", "amsterdam"),
    ("belgium", "capital", "brussels"),
    ("denmark", "capital", "copenhagen"),
    ("scotland", "capital", "edinburgh"),

    # --- chemistry: element symbols (query-natural: "gold symbol" -> Au) -
    ("hydrogen", "symbol", "H"),
    ("helium", "symbol", "He"),
    ("lithium", "symbol", "Li"),
    ("carbon", "symbol", "C"),
    ("nitrogen", "symbol", "N"),
    ("oxygen", "symbol", "O"),
    ("fluorine", "symbol", "F"),
    ("neon", "symbol", "Ne"),
    ("sodium", "symbol", "Na"),
    ("aluminium", "symbol", "Al"),
    ("silicon", "symbol", "Si"),
    ("chlorine", "symbol", "Cl"),
    ("argon", "symbol", "Ar"),
    ("iron", "symbol", "Fe"),
    ("copper", "symbol", "Cu"),
    ("zinc", "symbol", "Zn"),
    ("silver", "symbol", "Ag"),
    ("tin", "symbol", "Sn"),
    ("mercury", "symbol", "Hg"),
    ("lead", "symbol", "Pb"),
    ("gold", "symbol", "Au"),
    ("uranium", "symbol", "U"),

    # --- science -------------------------------------------------------
    ("earth", "orbits", "sun"),
    ("moon", "orbits", "earth"),
    ("mars", "orbits", "sun"),
    ("light", "travels-at", "299792458-m-per-s"),
    ("water", "freezes-at", "0-celsius"),
    ("water", "boils-at", "100-celsius"),
    ("absolute-zero", "is", "minus-273.15-celsius"),
    ("pi", "approx", "3.14159"),
    ("e", "approx", "2.71828"),
    ("dna", "encodes", "genes"),
    ("gene", "is-made-of", "dna"),

    # --- computing -----------------------------------------------------
    ("python", "is-a", "programming-language"),
    ("c", "is-a", "programming-language"),
    ("linux", "is-a", "operating-system"),
    ("unix", "is-a", "operating-system"),
    ("python", "created-by", "guido-van-rossum"),
    ("linux", "created-by", "linus-torvalds"),
    ("unix", "created-by", "ken-thompson"),
    ("c", "created-by", "dennis-ritchie"),
    ("javascript", "created-by", "brendan-eich"),
    ("java", "created-by", "james-gosling"),
    ("www", "invented-by", "tim-berners-lee"),
    ("tcp-ip", "invented-by", "vint-cerf"),

    # --- history / people ---------------------------------------------
    ("einstein", "developed", "relativity"),
    ("darwin", "developed", "evolution"),
    ("newton", "developed", "calculus"),
    ("curie", "discovered", "radium"),
    ("galileo", "observed", "jupiter-moons"),
    ("turing", "designed", "turing-machine"),
    ("shakespeare", "wrote", "hamlet"),
    ("homer", "wrote", "odyssey"),
    ("beethoven", "composed", "ninth-symphony"),

    # --- reasoning chains (multi-hop path walking) ---------------------
    # teaches: socrates -> plato -> aristotle -> alexander
    ("socrates", "teaches", "plato"),
    ("plato", "teaches", "aristotle"),
    ("aristotle", "teaches", "alexander"),
    # is-a: socrates -> human -> mammal -> animal -> organism
    ("socrates", "is-a", "human"),
    ("human", "is-a", "mammal"),
    ("dog", "is-a", "mammal"),
    ("cat", "is-a", "mammal"),
    ("mammal", "is-a", "animal"),
    ("salmon", "is-a", "fish"),
    ("fish", "is-a", "animal"),
    ("sparrow", "is-a", "bird"),
    ("bird", "is-a", "animal"),
    ("animal", "is-a", "organism"),
    ("oak", "is-a", "tree"),
    ("tree", "is-a", "organism"),
    # knows: alice -> bob -> carol -> dave -> erin
    ("alice", "knows", "bob"),
    ("bob", "knows", "carol"),
    ("carol", "knows", "dave"),
    ("dave", "knows", "erin"),
    # multi-valued relation: one query, two true answers (designed ambiguity)
    ("jupiter", "has-moon", "io"),
    ("jupiter", "has-moon", "europa"),
]

# Relations whose queries encode TWO object tokens (n_<a> + n_<b>) plus the
# relation channel; used by the arithmetic compilation in the engine.
BINARY_RELATIONS = ("plus", "minus", "times")

# --- tool / action determination --------------------------------------
# The same machinery that determines a fact determines "the next X" in an
# agent loop: which tool to call, and what the tool needs next. These are
# walked as a chain: intent -> tool -> required argument -> argument type.
TOOL_FACTS: list[tuple[str, str, str]] = [
    ("ask-weather", "calls", "tool-get-weather"),
    ("tool-get-weather", "needs", "city-arg"),
    ("city-arg", "type", "string"),
    ("ask-time", "calls", "tool-current-time"),
    ("tool-current-time", "needs", "timezone-arg"),
    ("timezone-arg", "type", "string"),
    ("calculate", "calls", "tool-calculator"),
    ("tool-calculator", "needs", "expression-arg"),
    ("expression-arg", "type", "string"),
    ("search-web", "calls", "tool-web-search"),
    ("tool-web-search", "needs", "query-arg"),
    ("query-arg", "type", "string"),
    ("read-file", "calls", "tool-file-read"),
    ("tool-file-read", "needs", "path-arg"),
    ("path-arg", "type", "string"),
    ("send-message", "calls", "tool-send-message"),
    ("tool-send-message", "needs", "recipient-arg"),
    ("recipient-arg", "type", "string"),
    ("translate", "calls", "tool-translate"),
    ("tool-translate", "needs", "text-arg"),
    ("text-arg", "type", "string"),
    ("book-flight", "calls", "tool-booking"),
    ("tool-booking", "needs", "date-arg"),
    ("date-arg", "type", "string"),
]
