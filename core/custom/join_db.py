import os, shutil
from sheduler.start_create_models import connect, init_db

journal_dir = '/home/neuro_linux/data/models/'
union_dir = '/home/neuro_linux/data/union_models/'

connect("""
    CREATE TABLE IF NOT EXISTS journal 
    (id INTEGER PRIMARY KEY AUTOINCREMENT, y_pred REAL, all_pred REAL, y_succses REAL, 
     amount REAL, y_test REAL, model TEXT, alpha REAL, gamma REAL, alpha_fp REAL)""",
     file_db=union_dir+'journal.db')
try:
    os.mkdir(union_dir)
except: pass
# lst = [x for x in os.scandir(journal_dir)]
lst_dir = [journal_dir+x.name for x in os.scandir(journal_dir) if x.is_dir()]
num_model = 0
dict_files = {}
for dir_ in lst_dir:
    print(dir_+'/journal.db')
    recs = connect("SELECT * FROM journal", file_db=dir_+'/journal.db')
    j = 0
    for rec in recs:
        old_file = dir_ + '/' + rec[6]+'.keras'
        new_file = F"{union_dir}{num_model}_{j}.keras"
        shutil.copy(old_file, new_file)
        print(rec, old_file, new_file)
        rec = list(rec)
        rec[6] = new_file.split('/')[-1]
        print(rec)
        connect( """INSERT INTO journal (y_pred, all_pred, y_succses, 
     amount, y_test, model, alpha, gamma, alpha_fp) VALUES(?,?,?,?,?,?,?,?,?); """,
                 file_db=union_dir+'journal.db', param=rec[1:])
        j += 1
    num_model += 1
    # for file in os.scandir(dir_+'journal.db') if file.

