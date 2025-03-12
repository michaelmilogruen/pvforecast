import psycopg2
from psycopg2.extras import RealDictCursor

def connect_to_db():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="InfraTrans_c",
            user="postgres",
            password="password"
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def list_tables():
    """List all tables in the database."""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cur.fetchall()
            print("\nTables in database:")
            for table in tables:
                print(f"- {table['table_name']}")
    except Exception as e:
        print(f"Error listing tables: {e}")
    finally:
        conn.close()

def describe_table(table_name):
    """Show the structure of a specific table."""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = %s
            """, (table_name,))
            columns = cur.fetchall()
            print(f"\nStructure of table '{table_name}':")
            for col in columns:
                length = f"({col['character_maximum_length']})" if col['character_maximum_length'] else ""
                print(f"- {col['column_name']}: {col['data_type']}{length}")
    except Exception as e:
        print(f"Error describing table: {e}")
    finally:
        conn.close()

def execute_query(query, params=None):
    """Execute a custom SQL query."""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if query.strip().lower().startswith('select'):
                results = cur.fetchall()
                print("\nQuery results:")
                for row in results:
                    print(row)
            else:
                conn.commit()
                print("Query executed successfully")
    except Exception as e:
        print(f"Error executing query: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    while True:
        print("\nPostgreSQL Database Access")
        print("1. List all tables")
        print("2. Describe a table")
        print("3. Execute custom query")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            list_tables()
        elif choice == '2':
            table_name = input("Enter table name: ")
            describe_table(table_name)
        elif choice == '3':
            query = input("Enter SQL query: ")
            execute_query(query)
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")