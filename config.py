import os

class Config:
    # Veri yolları
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    
    # Model parametreleri
    IMG_SIZE = (32, 32)
    BATCH_SIZE = 64
    EPOCHS = 50
    NUM_CLASSES = 44  # 44 farklı Osmanlıca karakter
    
    # Etiket eşlemesi
    LABEL_MAP = {
    1: 'elif',
    2: 'be',
    3: 'te',
    4: 'se',
    5: 'cim',
    6: 'ha',
    7: 'hı',
    8: 'dal',
    9: 'zel',
    10: 'ra',
    11: 'ze',
    12: 'sin',
    13: 'şın',
    14: 'şad',
    15: 'dad',
    16: 'tı',
    17: 'zı',
    18: 'ayn',
    19: 'ğayn',
    20: 'fe',
    21: 'kaf',
    22: 'kef',
    23: 'lam',
    24: 'mim',
    25: 'nun',
    26: 'he',
    27: 'vav',
    28: 'ye',
    29: 'pe',
    30: 'çim',
    31: 'je',
    32: 'gef',
    33: 'nef',
    34: '1',
    35: '2',
    36: '3',
    37: '4',
    38: '5',
    39: '6',
    40: '7',
    41: '8',
    42: '9',
    43: '0',
    44: 'lamelif'
}

    
    # Kayıt yolları
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_models', 'ottoman_ocr_model.h5')