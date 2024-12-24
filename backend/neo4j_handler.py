from neo4j import GraphDatabase, exceptions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import traceback
from datetime import datetime
import backoff


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Neo4jHandler:
    def __init__(self, uri, max_retry_time=30):
        self.uri = uri
        self.driver = None
        self.max_retry_time = max_retry_time
        self.connect()

    @backoff.on_exception(backoff.expo, 
                         (exceptions.ServiceUnavailable, 
                          exceptions.SessionExpired), 
                         max_time=30)
    
    def connect(self):
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(self.uri)
                self.check_connection()
            except Exception as e:
                logging.error(f"Failed to connect to Neo4j: {str(e)}")
                raise

    def ensure_connection(self):
        """Ensures there's a valid connection before operations"""
        try:
            if self.driver is None:
                self.connect()
            self.check_connection()
        except Exception:
            self.close()
            self.connect()

    def check_connection(self):
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
                logging.info("Successfully connected to Neo4j!")
        except Exception as e:
            logging.error(f"Connection check failed: {str(e)}")
            self.close()
            raise

    def close(self):
        if self.driver:
            try:
                self.driver.close()
            except Exception as e:
                logging.error(f"Error closing connection: {str(e)}")
            finally:
                self.driver = None


    def create_user(self, user_id, name):
        self.ensure_connection()
        logger.debug(f"Creating user with ID: {user_id}, Name: {name}")
        
        query = """
        MERGE (u:User {user_id: $user_id})
        ON CREATE SET u.name = $name
        ON MATCH SET u.name = $name
        RETURN u
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {
                    "user_id": user_id,
                    "name": name
                })
                record = result.single()
                if record:
                    logger.info(f"Successfully created/updated user: {user_id}")
                    return record
                else:
                    logger.error(f"Failed to create user: {user_id}")
                    logger.error(traceback.format_exc())
                    raise Exception("Failed to create user")
                    
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_post(self, user_id, content):
        self.ensure_connection()
        logger.debug(f"Creating post for user {user_id} with content: {content}")
        sentiment = self._analyze_sentiment(content)
    
        # Use Neo4j's datetime() function instead of Python's datetime
        query = """
        MATCH (u:User {user_id: $user_id})
        CREATE (p:Post {
            post_id: randomUUID(),
            content: $content,
            sentiment: $sentiment,
            timestamp: datetime()
        })-[:POSTED_BY]->(u)
        WITH p
        SET p.timestamp = toString(p.timestamp)
        RETURN p
        """
    
        try:
            with self.driver.session() as session:
                result = session.run(query, 
                    user_id=user_id,
                    content=content,
                    sentiment=sentiment
                )
                record = result.single()
                if record is None:
                    raise Exception(f"User with ID {user_id} not found")
            
                self._update_relationships()
                return {"p": record["p"]}
            
        except Exception as e:
            logger.error(f"Error creating post: {str(e)}")
            raise

    def get_user_posts(self, user_id):
        self.ensure_connection()
        query = """
        MATCH (p:Post)-[:POSTED_BY]->(u:User {user_id: $user_id})
        RETURN p.post_id AS post_id, 
               p.content AS content, 
               p.sentiment AS sentiment, 
               p.timestamp AS timestamp
        ORDER BY p.timestamp DESC
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, user_id=user_id)
                posts = [dict(record) for record in result]
                return posts
        except Exception as e:
            logger.error(f"Error getting user posts: {str(e)}")
            raise


    def _update_relationships(self):
        try:
            # Get all users and their posts
            query = """
            MATCH (u:User)
            OPTIONAL MATCH (u)<-[:POSTED_BY]-(p:Post)
            WITH u, COLLECT(p.content) AS contents
            WHERE size(contents) > 0
            RETURN u.user_id AS user_id, contents
            """
        
            with self.driver.session() as session:
                result = session.run(query)
                users_data = [dict(record) for record in result]
        
            if len(users_data) < 2:
                logger.debug("Not enough users with posts to create relationships")
                return
        
            # Create user content vectors
            user_contents = {
                user['user_id']: ' '.join(filter(None, user['contents'])) 
                for user in users_data
            }
        
            # Calculate similarities
            content_matrix = self.vectorizer.fit_transform(user_contents.values())
            similarity_matrix = cosine_similarity(content_matrix)
        
            # Create relationships
            user_ids = list(user_contents.keys())
            threshold = 0.05 # Adjust this value to make relationships more or less strict
        
            # Clear existing relationships
            self._clear_relationships()
        
            # Create new relationships
            relationships = []
            for i in range(len(user_ids)):
                for j in range(len(user_ids)):
                    if i != j and similarity_matrix[i][j] > threshold:
                        relationships.append((user_ids[i], user_ids[j]))
        
            if relationships:
                self._create_relationships(relationships)
                logger.info(f"Created {len(relationships)} relationships")
            else:
                logger.info("No relationships met the similarity threshold")
            
        except Exception as e:
            logger.error(f"Error updating relationships: {str(e)}")
            logger.error(traceback.format_exc())

    def _clear_relationships(self):
        query = "MATCH ()-[r:SIMILAR_CONTENT]->() DELETE r"
        with self.driver.session() as session:
            session.run(query)
    
    def _create_relationships(self, relationships):
        query = """
        UNWIND $relationships AS rel
        MATCH (u1:User {user_id: rel[0]})
        MATCH (u2:User {user_id: rel[1]})
        MERGE (u1)-[r:SIMILAR_CONTENT]->(u2)
        """
        with self.driver.session() as session:
            session.run(query, {"relationships": relationships})


    def get_all_users(self):
        self.ensure_connection()
        query = """
        MATCH (u:User)
        RETURN u.user_id AS user_id, u.name AS name
        ORDER BY u.name
        """
        try:
            with self.driver.session() as session:
                result = session.run(query)
                users = [dict(record) for record in result]
                logger.debug(f"Retrieved users: {users}")
                return users
        except Exception as e:
            logger.error(f"Error getting users: {str(e)}")
            raise

    def get_user_posts(self, user_id):
        self.ensure_connection()
        query = """
        MATCH (p:Post)-[:POSTED_BY]->(u:User {user_id: $user_id})
        RETURN p.post_id AS post_id, p.content AS content, 
               p.sentiment AS sentiment, p.timestamp AS timestamp
        ORDER BY p.timestamp DESC
        """
        with self.driver.session() as session:
            result = session.run(query, {"user_id": user_id})
            return [dict(record) for record in result]

    def _analyze_sentiment(self, content):
        try:
            # Enhanced sentiment analysis with emoji support
            emoji_sentiments = {
            '😊': 1, '😄': 1, '😃': 1, '😍': 1, '❤️': 1, '👍': 1,  # Positive
            '😢': -1, '😭': -1, '😡': -1, '😠': -1, '💔': -1, '👎': -1  # Negative
            }
    
            # Extract words and emojis
            content_lower = content.lower()
            words = content_lower.split()

            positive_words={
    'joyful', 'outstanding', 'marvelous', 'spectacular', 'cheerful', 
    'blissful', 'ecstatic', 'content', 'grateful', 'pleased', 
    'satisfied', 'bright', 'optimistic', 'hopeful', 'jubilant',
    'incredible', 'remarkable', 'charming', 'gracious', 'serene','loves','good', 'great', 'awesome', 'excellent', 'happy', 'love', 'wonderful',
                'fantastic', 'delighted', 'brilliant', 'amazing'
}
            negative_words = {
                'miserable', 'dreadful', 'frustrated', 'pathetic', 'grim', 
    'depressed', 'hopeless', 'unbearable', 'gloomy', 'angst',
    'distressed', 'regretful', 'melancholy', 'tragic', 'devastated',
    'infuriated', 'vengeful', 'resentful', 'bitter', 'disheartened','hates','bad', 'terrible', 'awful', 'horrible', 'sad', 'hate', 'disappointed',
                'poor', 'angry', 'upset', 'unhappy'
            }
            
            words = content.lower().split()
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            # Add emoji sentiment
            for char in content:
                if char in emoji_sentiments:
                    if emoji_sentiments[char] > 0:
                        pos_count += 1
                    else:
                        neg_count += 1
                        
            if pos_count > neg_count:
                return "positive"
            elif neg_count > pos_count:
                return "negative"
            return "neutral"
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return "neutral"  # Default to neutral if analysis fails
   
    def get_recent_posts(self):
        self.ensure_connection()
        query = """
        MATCH (p:Post)-[:POSTED_BY]->(u:User)
        WHERE p.deleted is NULL
        RETURN p.post_id AS post_id,
            p.content AS content,
            p.sentiment AS sentiment,
            p.timestamp AS timestamp,
            u.name AS user_name,
            u.user_id AS user_id
        ORDER BY p.timestamp DESC
        LIMIT 10
        """
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error getting recent posts: {str(e)}")
            raise
    
    def soft_delete_post(self, post_id):
        query = """
        MATCH (p:Post {post_id: $post_id})
        SET p.deleted = true,
            p.deleted_at = datetime()
        RETURN p
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, post_id=post_id)
                record = result.single()
                if not record:
                    raise Exception("Post not found")
                return record
        except Exception as e:
            logger.error(f"Error deleting post: {str(e)}")
            raise
    
    def restore_post(self, post_id):
        query = """
        MATCH (p:Post {post_id: $post_id})
        REMOVE p.deleted,p.deleted_at
        RETURN p
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, post_id=post_id)
                record = result.single()
                if not record:
                    raise Exception("Post not found")
                return record
        except Exception as e:
            logger.error(f"Error restoring post: {str(e)}")
            raise
    