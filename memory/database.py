"""
SQLite database operations for shared memory across agents.
"""
import os
import sqlite3 as sqlite
import logging
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path

from .models import ProcessingRecord, CREATE_TABLES_SQL
from config import DATABASE_URL, SQLITE_CONFIG

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations for the multi-agent system.
    """
    
    def __init__(self, db_path: Path = DATABASE_URL):
        self.db_path = db_path
        self.init_database()


    def init_database(self):
        """Initialize the database with required tables."""
        try:
            # Optional: check if file exists but is not a valid SQLite file
            if os.path.exists(self.db_path) and os.path.getsize(self.db_path) < 100:
                logger.warning(f"Deleting invalid DB file: {self.db_path}")
                os.remove(self.db_path)

            with self.get_connection() as conn:
                conn.executescript(CREATE_TABLES_SQL)
                conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

        except sqlite.DatabaseError as db_err:
            logger.error(f"DatabaseError: Possibly corrupted file: {self.db_path}. Error: {db_err}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing database: {e}")
            raise

    
    # def init_database(self):
    #     """Initialize the database with required tables."""
    #     try:
    #         with self.get_connection() as conn:
    #             conn.executescript(CREATE_TABLES_SQL)
    #             conn.commit()
    #         logger.info(f"Database initialized at {self.db_path}")
    #     except Exception as e:
    #         logger.error(f"Failed to initialize database: {e}")
    #         raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        try:
            conn = sqlite.connect(
                self.db_path,
                check_same_thread=SQLITE_CONFIG["check_same_thread"],
                timeout=SQLITE_CONFIG["timeout"],
                isolation_level=SQLITE_CONFIG["isolation_level"]
            )
            conn.row_factory = sqlite.Row
            yield conn
        except sqlite.DatabaseError as e:
            logger.error(f"Failed to connect to database at {self.db_path}: {e}")
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass


    # @contextmanager
    # def get_connection(self):
    #     """Context manager for database connections."""
    #     conn = sqlite.connect(
    #         self.db_path,
    #         check_same_thread=SQLITE_CONFIG["check_same_thread"],
    #         timeout=SQLITE_CONFIG["timeout"],
    #         isolation_level=SQLITE_CONFIG["isolation_level"]
    #     )
    #     conn.row_factory = sqlite.Row  # Enable column access by name
    #     try:
    #         yield conn
    #     finally:
    #         conn.close()
    
    def create_record(self, record: ProcessingRecord) -> int:
        """
        Insert a new processing record and return its ID.
        """
        try:
            with self.get_connection() as conn:
                record_dict = record.to_dict()
                # Remove id for insertion
                record_dict.pop('id', None)
                
                columns = ', '.join(record_dict.keys())
                placeholders = ', '.join(['?' for _ in record_dict])
                values = list(record_dict.values())
                
                cursor = conn.execute(
                    f"INSERT INTO processing_records ({columns}) VALUES ({placeholders})",
                    values
                )
                record_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Created record with ID: {record_id}")
                return record_id
                
        except Exception as e:
            logger.error(f"Failed to create record: {e}")
            raise
    
    def get_record(self, record_id: int) -> Optional[ProcessingRecord]:
        """
        Retrieve a processing record by ID.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM processing_records WHERE id = ?",
                    (record_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return ProcessingRecord.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get record {record_id}: {e}")
            raise
    
    def update_record(self, record: ProcessingRecord) -> bool:
        """
        Update an existing processing record.
        """
        if not record.id:
            raise ValueError("Record must have an ID to update")
        
        try:
            with self.get_connection() as conn:
                record_dict = record.to_dict()
                record_id = record_dict.pop('id')
                
                set_clause = ', '.join([f"{key} = ?" for key in record_dict.keys()])
                values = list(record_dict.values()) + [record_id]
                
                cursor = conn.execute(
                    f"UPDATE processing_records SET {set_clause} WHERE id = ?",
                    values
                )
                conn.commit()
                
                updated = cursor.rowcount > 0
                if updated:
                    logger.info(f"Updated record {record_id}")
                else:
                    logger.warning(f"No record found with ID {record_id}")
                
                return updated
                
        except Exception as e:
            logger.error(f"Failed to update record {record.id}: {e}")
            raise
    
    def get_total_records(self) -> int:
        """
        Returns the total number of records in the processing table.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM processing_records")
                total = cursor.fetchone()[0]
                return total
        except Exception as e:
            raise RuntimeError(f"Failed to get total records: {e}")



    def get_records_by_status(self, status: str) -> List[ProcessingRecord]:
        """
        Retrieve all records with a specific status.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM processing_records WHERE status = ? ORDER BY timestamp DESC",
                    (status,)
                )
                rows = cursor.fetchall()
                
                return [ProcessingRecord.from_dict(dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get records by status {status}: {e}")
            raise
    
    def get_records_by_thread(self, thread_id: str) -> List[ProcessingRecord]:
        """
        Retrieve all records belonging to a thread.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM processing_records WHERE thread_id = ? ORDER BY timestamp ASC",
                    (thread_id,)
                )
                rows = cursor.fetchall()
                
                return [ProcessingRecord.from_dict(dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get records by thread {thread_id}: {e}")
            raise
    
    def get_recent_records(self, limit: int = 50) -> List[ProcessingRecord]:
        """
        Retrieve the most recent processing records.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM processing_records ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
                rows = cursor.fetchall()
                
                return [ProcessingRecord.from_dict(dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get recent records: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics from the database.
        """
        try:
            with self.get_connection() as conn:
                stats = {}
                
                # Total records
                cursor = conn.execute("SELECT COUNT(*) as count FROM processing_records")
                stats['total_records'] = cursor.fetchone()['count']
                
                # Records by status
                cursor = conn.execute("""
                    SELECT status, COUNT(*) as count 
                    FROM processing_records 
                    GROUP BY status
                """)
                stats['by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}
                
                # Records by format
                cursor = conn.execute("""
                    SELECT format, COUNT(*) as count 
                    FROM processing_records 
                    GROUP BY format
                """)
                stats['by_format'] = {row['format']: row['count'] for row in cursor.fetchall()}
                
                # Records by intent
                cursor = conn.execute("""
                    SELECT intent, COUNT(*) as count 
                    FROM processing_records 
                    GROUP BY intent
                """)
                stats['by_intent'] = {row['intent']: row['count'] for row in cursor.fetchall()}
                
                # Average processing time
                cursor = conn.execute("""
                    SELECT AVG(processing_time) as avg_time 
                    FROM processing_records 
                    WHERE processing_time > 0
                """)
                avg_time = cursor.fetchone()['avg_time']
                stats['avg_processing_time'] = round(avg_time, 2) if avg_time else 0
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
    
    def search_records(self, query: str, limit: int = 20) -> List[ProcessingRecord]:
        """
        Search records by source, intent, or extracted content.
        """
        try:
            with self.get_connection() as conn:
                search_query = f"%{query}%"
                cursor = conn.execute("""
                    SELECT * FROM processing_records 
                    WHERE source LIKE ? 
                       OR intent LIKE ? 
                       OR extracted_fields LIKE ?
                       OR sender LIKE ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (search_query, search_query, search_query, search_query, limit))
                
                rows = cursor.fetchall()
                return [ProcessingRecord.from_dict(dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to search records: {e}")
            raise
    
    def delete_record(self, record_id: int) -> bool:
        """
        Delete a processing record by ID.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM processing_records WHERE id = ?",
                    (record_id,)
                )
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"Deleted record {record_id}")
                else:
                    logger.warning(f"No record found with ID {record_id}")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete record {record_id}: {e}")
            raise
    
    def clear_database(self) -> bool:
        """
        Clear all records from the database (use with caution).
        """
        try:
            with self.get_connection() as conn:
                conn.execute("DELETE FROM processing_records")
                conn.commit()
                logger.warning("All records cleared from database")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise


# Global database instance
db_manager = DatabaseManager()