import cv2

SCORE_THRESHOLD = 0.5
ISLET_LABEL = 0
EXO_LABEL = 1

# visualization constants
CONTOUR_THICKNESS = 1
BBOX_THICKNESS = 1

BBOX_SCORE_FONT = cv2.FONT_HERSHEY_SIMPLEX
BBOX_SCORE_SCALE = 0.3
BBOX_SCORE_THICKNESS = 1
BBOX_SCORE_BG_COLOR = (0, 0, 0)
BBOX_SCORE_FONT_COLOR = (255, 255, 255)

INSTANCE_MASK_OPACITY = 0.5

# mask diff
FALSE_NEGATIVE_COLOR = (250, 0, 50)
FALSE_POSITIVE_COLOR = (100, 100, 255)
MATCHED_COLOR = (50, 255, 50)

# colors
CONTOUR_COLORS_RGB = [
    (234, 221, 202),  # Almond
    (255, 191, 0),    # Amber
    (251, 206, 177),  # Apricot
    (0, 255, 255),    # Aqua
    (137, 207, 240),  # Baby blue
    (8, 143, 143),    # Blue green
    (225, 193, 110),  # Brass
    (0, 150, 255),    # Bright blue
    (170, 255, 0),    # Bright green
    (255, 172, 28),   # Bright orange
    (191, 64, 191),   # Bright purple
    (255, 234, 0),    # Bright yellow
    (205, 127, 50),   # Bronze
    (165, 42, 42),    # Brown
    (218, 160, 109),  # Buff
    (242, 140, 40),   # Cadmium orange
    (193, 154, 107),  # Camel
    (255, 255, 143),  # Canary yellow
    (175, 225, 175),  # Celadon
    (255, 127, 80),   # Coral
    (248, 131, 121),  # Coral pink
    (100, 149, 237),  # Cornflower blue
    (223, 255, 0),    # Chartreuse
    (228, 208, 10),   # Citrine
    (255, 253, 208),  # Cream
    (169, 169, 169),  # Dark gray
    (170, 51, 106),   # Dark pink
    (111, 143, 175),  # Denim
    (250, 213, 165),  # Desert
    (201, 169, 166),  # Dusty rose
    (80, 200, 120),   # Emerald green
    (229, 170, 112),  # Fawn
    (34, 139, 34),    # Forest green
    (255, 192, 0),    # Golden yellow
    (124, 252, 0),    # Grass green
    (255, 105, 180),  # Hot pink
    (0, 163, 108),    # Jade
    (42, 170, 138),   # Jungle green
    (76, 187, 23),    # Kelly green
    (240, 230, 140),  # Khaki
    (230, 230, 250),  # Lavender
    (250, 250, 51),   # Lemon yellow
    (173, 216, 230),  # Light blue
    (196, 164, 132),  # Light brown
    (144, 238, 144),  # Light green
    (255, 213, 128),  # Light orange
    (255, 182, 193),  # Light pink
    (207, 159, 255),  # Light violet
    (255, 0, 255),    # Magenta
    (152, 251, 152),  # Mint green
    (31, 81, 255),    # Neon blue
    (15, 255, 80),    # Neon green
    (218, 112, 214),  # Orchid
    (250, 200, 152),  # Pastel orange
    (248, 200, 220),  # Pastel pink
    (250, 160, 160),  # Pastel red
    (195, 177, 225),  # Pastel purple
    (255, 250, 160),  # Pastel yellow
    (255, 229, 180),  # Peach
    (201, 204, 63),   # Pear
    (180, 196, 36),   # Peridot
    (204, 204, 255),  # Periwinkle
    (248, 152, 128),  # Pink orange
    (150, 222, 209),  # Robin egg blue
    (65, 105, 225),   # Royal blue
    (250, 128, 114),  # Salmon
    (159, 226, 191),  # Seafoam green
    (255, 245, 238),  # Seashell
    (160, 82, 45),    # Sienna
    (192, 192, 192),  # Silver
    (135, 206, 235),  # Sky blue
    (70, 130, 180),   # Steel blue
    (0, 128, 128),    # Teal
    (64, 224, 208),   # Turqoise
    (64, 181, 173),   # Verdigris
    (127, 0, 255),    # Violet
    (227, 115, 131),  # Watermelon pink
]