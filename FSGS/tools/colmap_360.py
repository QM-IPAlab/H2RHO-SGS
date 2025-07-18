import os
import numpy as np
import sys
import sqlite3


IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def pipeline(scene, base_path, n_views):
    llffhold = 8
    view_path = str(n_views) + '_views'
    os.chdir(base_path + scene)
    os.system('rm -r ' + view_path)
    os.mkdir(view_path)
    os.chdir(view_path)
    scen_path_full = base_path + scene
    view_path_full = base_path + scene + '/'+view_path
    print(f"-------------------------------{view_path_full}--------------{scen_path_full}")
    os.mkdir('created')
    os.mkdir('triangulated')
    os.mkdir('images')
    os.system(f'singularity run --nv /home/e/eez095/project/FSGS/colmap_latest.sif colmap model_converter --input_path {scen_path_full}/sparse/0/ --output_path {scen_path_full}/sparse/0/  --output_type TXT')


    images = {}
    with open(f'{scen_path_full}/sparse/0/images.txt', "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8]) 
                image_name = elems[9] 
                fid.readline().split() 
                images[image_name] = elems[1:] 
    
    img_list = sorted(images.keys(), key=lambda x: x)
    train_img_list = [c for idx, c in enumerate(img_list) if idx % llffhold != 0]
    if n_views > 0:
        idx_sub = [round_python3(i) for i in np.linspace(0, len(train_img_list)-1, n_views)]
        train_img_list = [c for idx, c in enumerate(train_img_list) if idx in idx_sub]


    for img_name in train_img_list:
        os.system(f'cp {scen_path_full}/images/' + img_name + f'{view_path_full}/images/' + img_name)

    os.system(f'cp {scen_path_full}/sparse/0/cameras.txt created/.')
    with open(f'{view_path_full}/created/points3D.txt', "w") as fid:
        pass
    
    res = os.popen( f'singularity run --nv /home/e/eez095/project/FSGS/colmap_latest.sif colmap feature_extractor --database_path {view_path_full}/database.db --image_path {view_path_full}/images  --SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1').read()
    os.system( f'singularity run --nv /home/e/eez095/project/FSGS/colmap_latest.sif colmap exhaustive_matcher --database_path {view_path_full}/database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768')
    db = COLMAPDatabase.connect(f'{view_path_full}/database.db') # 这表是空的
    db_images = db.execute("SELECT * FROM images")
    img_rank = [db_image[1] for db_image in db_images]
    print(img_rank, res)
    with open(f'{view_path_full}/created/images.txt', "w") as fid:
        for idx, img_name in enumerate(img_rank):
            print(img_name)
            data = [str(1 + idx)] + [' ' + item for item in images[os.path.basename(img_name)]] + ['\n\n']
            fid.writelines(data)

    os.system(f'singularity run --nv /home/e/eez095/project/FSGS/colmap_latest.sif colmap point_triangulator --database_path {view_path_full}/database.db --image_path {view_path_full}/images --input_path {view_path_full}/created  --output_path {view_path_full}/triangulated  --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100')
    os.system(f'singularity run --nv /home/e/eez095/project/FSGS/colmap_latest.sif colmap model_converter  --input_path {view_path_full}/triangulated --output_path {view_path_full}/triangulated  --output_type TXT')
    os.system(f'singularity run --nv /home/e/eez095/project/FSGS/colmap_latest.sif colmap image_undistorter --image_path {view_path_full}/images --input_path {view_path_full}/triangulated --output_path {view_path_full}/dense')
    os.system(f'singularity run --nv /home/e/eez095/project/FSGS/colmap_latest.sif colmap patch_match_stereo --workspace_path dense')
    os.system(f'colmap stereo_fusion --workspace_path {view_path_full}/dense --output_path {view_path_full}/dense/fused.ply')


for scene in ['bicycle', 'bonsai', 'counter', 'garden',  'kitchen', 'room', 'stump']:
    pipeline(scene, base_path = '/home/e/eez095/project/FSGS/dataset/mipnerf360/', n_views = 24)  # please use absolute path!


