from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from neo4j_handler import Neo4jHandler
from datetime import datetime
import traceback
import logging

#Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

try:
    neo4j_handler = Neo4jHandler("bolt://localhost:7687")  # Add your Neo4j credentials
except Exception as e:
    logger.error(f"Failed to initialize Neo4j connection: {str(e)}")
    raise

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route("/api/users", methods=["GET"])
def get_users():
    try:
        users = neo4j_handler.get_all_users()
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/users", methods=["POST"])
def add_user():
    try:
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        logger.debug(f"Received user data: {data}")
        
        # Validate required fields
        if not all(key in data for key in ["user_id", "name"]):
            logger.error("Missing required fields")
            return jsonify({"error": "Missing required fields: user_id and name"}), 400
        
        # Create user
        result = neo4j_handler.create_user(data["user_id"], data["name"])
        logger.debug(f"User creation result: {result}")
        
        if result and hasattr(result, "get"):
            user_data = {"user_id": data["user_id"], "name": data["name"]}
            logger.info(f"Successfully created user: {user_data}")
            return jsonify({"success": True, "user": user_data})
        else:
            logger.error("User creation failed")
            return jsonify({"error": "Failed to create user"}), 500
            
    except Exception as e:
        logger.error(f"Error in add_user: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/posts", methods=["POST"])
def add_post():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        logger.debug(f"Received post data: {data}")
        
        if not all(key in data for key in ["user_id", "content"]):
            return jsonify({"error": "Missing required fields: user_id and content"}), 400
        
        result = neo4j_handler.create_post(data["user_id"], data["content"])
        
        if result and "p" in result:
            post_dict = dict(result["p"])
            return jsonify({"success": True, "post": post_dict})
        else:
            logger.error("Post creation failed: No result returned")
            return jsonify({"error": "Failed to create post"}), 500
            
    except Exception as e:
        logger.error(f"Error in add_post: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/users/<user_id>/posts", methods=["GET"])
def get_user_posts(user_id):
    try:
        posts = neo4j_handler.get_user_posts(user_id)
        return jsonify(posts)
    except Exception as e:
        logger.error(f"Error getting posts: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/debug/connection", methods=["GET"])
def test_connection():
    try:
        with neo4j_handler.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            return jsonify({
                "status": "connected",
                "node_count": count,
                "message": "Successfully connected to Neo4j"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "Failed to connect to Neo4j"
        }), 500
    
#Delete

# Update other routes with better error handling
@app.route("/api/posts/<post_id>", methods=["DELETE"])
def delete_post(post_id):
    try:
        logger.debug(f"Attempting to delete post {post_id}")
        result = neo4j_handler.soft_delete_post(post_id)
        return jsonify({"success": True, "message": "Post deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting post: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/posts/<post_id>/restore", methods=["POST"])
def restore_post(post_id):
    try:
        logger.debug(f"Attempting to restore post {post_id}")
        result = neo4j_handler.restore_post(post_id)
        return jsonify({"success": True, "message": "Post restored successfully"})
    except Exception as e:
        logger.error(f"Error restoring post: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# Add to recent posts
@app.route("/api/recent-posts", methods=["GET"])
def get_recent_posts():
    try:
        posts = neo4j_handler.get_recent_posts()
        return jsonify(posts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/graph", methods=["GET"])
def get_graph():
    try:
        with neo4j_handler.driver.session() as session:
            # Get nodes with their sentiment
            node_query = """
            MATCH (u:User)
            OPTIONAL MATCH (u)<-[:POSTED_BY]-(p:Post)
            WITH u, COLLECT(p.content) as contents, COLLECT(p.sentiment) as sentiments
            RETURN u.user_id AS id, u.name AS name,
                   CASE
                     WHEN size(sentiments) = 0 THEN 'neutral'
                     WHEN size([s IN sentiments WHERE s = 'positive']) > size([s IN sentiments WHERE s = 'negative'])
                     THEN 'positive'
                     WHEN size([s IN sentiments WHERE s = 'negative']) > size([s IN sentiments WHERE s = 'positive'])
                     THEN 'negative'
                     ELSE 'neutral'
                   END AS sentiment,
                   contents
            """
            nodes = [dict(record) for record in session.run(node_query)]

            # Get predicted relationships (SIMILAR_CONTENT)
            edge_query = """
            MATCH (u1:User)-[r:SIMILAR_CONTENT]->(u2:User)
            RETURN u1.user_id AS source, u2.user_id AS target
            """
            edges = [dict(record) for record in session.run(edge_query)]

            return jsonify({"nodes": nodes, "edges": edges})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    

@app.teardown_appcontext
def close_connection(exception):
    neo4j_handler.close()

if __name__ == "__main__":
    app.run(debug=True, port=5001)