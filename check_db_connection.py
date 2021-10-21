import psycopg2

# conn = psycopg2.connect(
#     host="localhost",
#     database="rajnish_psf",
#     user="postgres",
#     password="")

# # #LOCAL for Rajnish
# GS_USER = "rajnish_bns"
# GS_DB = "local_psf"
# GS_PWD = "bns*789"
# GS_HOST = "localhost"


# GS_USER = "rajnish_bns"
# GS_DB = "rajnish_psf"
# GS_PWD = "bns*789"
# GS_HOST = "localhost"

GS_USER = "Heart_watch_user"
GS_DB = "Heart_watch_DB"
GS_PWD = "heartwatch*789"
GS_HOST = "localhost"


print("Connecting to database.......")
db_conn = psycopg2.connect(database=GS_DB, user=GS_USER, password=GS_PWD, host=GS_HOST, port=5432)
print("Connected.")

