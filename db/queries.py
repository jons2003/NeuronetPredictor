class Queries:
    col_matches = {'id': 'INTEGER PRIMARY KEY AUTOINCREMENT', 'date': 'TIMESTAMP', 'league': 'TEXT', 'host': 'TEXT',
               'guest': 'TEXT', 'h_res': 'INT', 'g_res': 'INT', 'b1': 'REAL', 'bX': 'REAL', 'b2': 'REAL', 'url': 'TEXT'}
    col_HA = {'HA_win': 'INT', 'HA_draw': 'INT', 'HA_loss': 'INT', 'HA_goal_avg': 'REAL', 'HA_loss_avg': 'REAL',
              'HA_yellow': 'REAL', 'HA_red': 'REAL', 'HA_corner': 'REAL', 'HA_penalty': 'REAL', 'HA_offside': 'REAL'}
    col_GA = {'GA_win': 'INT', 'GA_draw': 'INT', 'GA_loss': 'INT', 'GA_goal_avg': 'REAL', 'GA_loss_avg': 'REAL',
              'GA_yellow': 'REAL', 'GA_red': 'REAL', 'GA_corner': 'REAL', 'GA_penalty': 'REAL', 'GA_offside': 'REAL'}
    col_HJ = {'HJ_win': 'INT', 'HJ_draw': 'INT', 'HJ_loss': 'INT', 'HJ_goal_avg': 'REAL', 'HJ_loss_avg': 'REAL',
              'HJ_yellow': 'REAL', 'HJ_red': 'REAL', 'HJ_corner': 'REAL', 'HJ_penalty': 'REAL', 'HJ_offside': 'REAL'}
    col_GJ = {'GJ_win': 'INT', 'GJ_draw': 'INT', 'GJ_loss': 'INT', 'GJ_goal_avg': 'REAL', 'GJ_loss_avg': 'REAL',
              'GJ_yellow': 'REAL', 'GJ_red': 'REAL', 'GJ_corner': 'REAL', 'GJ_penalty': 'REAL', 'GJ_offside': 'REAL'}
    col_torn = {'tourn': 'TEXT'}
    columns = {**col_matches, **col_HA, **col_GA, **col_HJ, **col_GJ, **col_torn}

    read_matches = '''SELECT date, host,guest FROM matches'''
    list_dubl_match = '''SELECT host, guest, COUNT(*) FROM matches GROUP BY host, guest HAVING COUNT(*) > 1;'''
    del_dubl_match = '''DELETE FROM matches WHERE id NOT IN (SELECT MIN(id) FROM matches GROUP BY host, guest); '''



    matches_not_fill = """SELECT * FROM matches  WHERE url != ''"""

    create_matches = """CREATE TABLE IF NOT EXISTS matches (id INTEGER PRIMARY KEY AUTOINCREMENT, date TIMESTAMP, league TEXT, host TEXT, guest TEXT, h_res INT, g_res INT, b1 REAL, bX REAL, b2 REAL, url TEXT);"""
    add_match = """INSERT INTO matches (date, league, host, guest, h_res, g_res, b1, bX, b2, url) VALUES(?,?,?,?,?,?,?,?,?,?); """
    read_match = """SELECT * FROM matches WHERE id=:num"""

    create_hosts_all = """CREATE TABLE IF NOT EXISTS hosts_all (id INT, win INT, draw INT, loss INT, goal_avg REAL, loss_avg REAL, yellow REAL, red REAL, corner REAL, penalty REAL, offside REAL);"""
    add_hosts_all = """INSERT INTO  hosts_all VALUES(?,?,?,?,?,?,?,?,?,?,?); """
    is_stat_hosts_all = """SELECT * FROM hosts_all WHERE id=:number"""
    create_guests_all = """CREATE TABLE IF NOT EXISTS guests_all (id INT, win INT, draw INT, loss INT, goal_avg REAL, loss_avg REAL, yellow REAL, red REAL, corner REAL, penalty REAL, offside REAL);"""
    add_guests_all = """INSERT INTO  guests_all VALUES (?,?,?,?,?,?,?,?,?,?,?); """
    is_stat_guests_all = """SELECT * FROM guests_all WHERE id=:number"""

    create_hosts_join = """CREATE TABLE IF NOT EXISTS hosts_join (id INT, win INT, draw INT, loss INT, goal_avg REAL, loss_avg REAL, yellow REAL, red REAL, corner REAL, penalty REAL, offside REAL);"""
    add_hosts_join = """INSERT INTO  hosts_join VALUES(?,?,?,?,?,?,?,?,?,?,?); """
    is_stat_hosts_join = """SELECT * FROM hosts_join WHERE id=:number"""
    create_guests_join = """CREATE TABLE IF NOT EXISTS guests_join (id INT, win INT, draw INT, loss INT, goal_avg REAL, loss_avg REAL, yellow REAL, red REAL, corner REAL, penalty REAL, offside REAL);"""
    add_guests_join = """INSERT INTO  guests_join VALUES (?,?,?,?,?,?,?,?,?,?,?); """
    is_stat_guests_join = """SELECT * FROM guests_join WHERE id=:number"""

    create_tourn = """CREATE TABLE IF NOT EXISTS tourn (id INT, tab TEXT);"""
    add_tourn = """INSERT INTO  tourn VALUES (?,?); """
    is_tourn = """SELECT * FROM tourn WHERE id=:number"""


    del_match = """ DELETE FROM matches WHERE id=:number"""


    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS orders (time TIMESTAMP, c INT, s TEXT, q INT, p REAL, sp REAL, ot TEXT, ps TEXT, s_ TEXT, ap REAL, x_ TEXT);"""
    ADD_REC = """INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?); """
    FIND_ORDER = """SELECT * FROM orders WHERE c=:number AND x_ = 'NEW'"""
    READ_DATA = """SELECT * FROM orders """
    COUNT = """SELECT COUNT(*) FROM orders WHERE c=:number and x_=:status"""
    DEL_NUM = """ DELETE FROM orders WHERE c=:number"""
    DEL_ALL = """ DELETE FROM orders"""
    FILTER = """WITH t as (WITH tmp AS (SELECT *,row_number() over (partition by c order by time) AS Nasc  FROM orders) 
                SELECT time, c, s, q, p, sp, ot, ps, s_, ap , x_  FROM tmp WHERE time<> 'Вход в позицию'AND (Nasc=1 OR x_='FILLED' OR x_='CANCELED') ORDER BY c)
                SELECT * FROM t ORDER BY time"""
    FIND_ACTIVE_ORDER = """SELECT c FROM orders WHERE ps=:posside AND (ot ='TAKE_PROFIT_MARKET' OR ot='STOP_MARKET')"""



    # ************************************************************************************************************************
    CREATE_TABLE_ACTIVE = """CREATE TABLE IF NOT EXISTS active (time TIMESTAMP, c INT, s TEXT, q INT, p REAL, sp REAL, ot TEXT, ps TEXT, s_ TEXT, ap REAL, x_ TEXT);"""
    DEL_ALL_ACTIVE = """ DELETE FROM active"""


    # ************************************************************************************************************************
    CREATE_TABLE_TEMP = """CREATE TABLE IF NOT EXISTS temp (id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, c INT, s TEXT, q INT, 
                            p REAL, sp REAL, ot TEXT, ps TEXT, s_ TEXT, ap REAL, x_ TEXT, pos_LONG REAL , 
                            amount_LONG REAL, avg_LONG REAL, pos_SHORT REAL, amount_SHORT REAL, avg_SHORT REAL);"""
    RESET_AUTO_TEMP = """UPDATE sqlite_sequence SET seq=0 """
    ADD_REC_TEMP = """INSERT INTO temp(time, c, s, q, p, sp, ot, ps, s_, ap , x_, pos_LONG, amount_LONG, avg_LONG, pos_SHORT, amount_SHORT, avg_SHORT)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?); """
    DEL_ALL_TEMP = """ DELETE FROM temp"""
    SELECT_ALL_TEMP = """SELECT * FROM temp"""


    # ************************************************************************************************************************
    CREATE_TABLE_FINISH = """CREATE TABLE IF NOT EXISTS finish 
                            (id INTEGER , time TIMESTAMP, c INT, s TEXT, q INT, p REAL, sp REAL, ot TEXT, ps TEXT, s_ TEXT, 
                            ap REAL, x_ TEXT, pos_LONG REAL , amount_LONG REAL, avg_LONG REAL, pos_SHORT REAL, amount_SHORT REAL, avg_SHORT REAL);"""
    ADD_REC_FINISH = """INSERT INTO finish(id, time, c, s, q, p, sp, ot, ps, s_, ap , x_, 
                        pos_LONG, amount_LONG, avg_LONG, pos_SHORT, amount_SHORT, avg_SHORT) VALUES(?,?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?)"""
    DEL_ALL_FINISH = """ DELETE FROM finish"""
    SELECT_ALL_FINISH = """SELECT * FROM finish"""
    SELECT_LAST_FILLED = """SELECT * FROM finish WHERE id != '' AND x_='FILLED' ORDER BY id DESC LIMIT 1"""

