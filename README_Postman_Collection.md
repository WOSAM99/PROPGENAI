# RAG Document Interaction API - Postman Collection

This Postman collection contains all the API endpoints for your RAG-based document interaction system.

## How to Import

1. Open Postman
2. Click "Import" button
3. Drag and drop the `RAG_API_Collection.postman_collection.json` file
4. The collection will be imported with all endpoints organized in folders

## Collection Structure

### 🔐 Authentication
- **POST /auth/exchange-supabase-token** - Exchange Supabase token for backend JWT
- **POST /auth/token** - Direct JWT creation (deprecated, dev only)

### 👤 Profile Management (TO BE IMPLEMENTED)
- **POST /profiles** - Create new profile
- **GET /profiles** - List all user profiles
- **GET /profiles/{id}** - Get specific profile details
- **PUT /profiles/{id}** - Update profile
- **DELETE /profiles/{id}** - Delete profile
- **GET /profiles/{id}/documents** - Get documents for profile

### 📄 Document Management (EXISTING)
- **POST /doc/upload** - Upload PDF and create embeddings
- **POST /doc/delete** - Delete ChromaDB collection

### 💬 Chat & Query (EXISTING + TO BE EXTENDED)
- **POST /chat/query** - Query RAG pipeline with prompts
- **GET /chat/history/{profile_id}** - Get chat history (TO BE IMPLEMENTED)
- **DELETE /chat/history/{profile_id}** - Clear chat history (TO BE IMPLEMENTED)

### ⚙️ System Prompts (TO BE IMPLEMENTED)
- **POST /system-prompts** - Create system prompt
- **GET /system-prompts** - List all system prompts
- **PUT /system-prompts/{id}** - Update system prompt
- **DELETE /system-prompts/{id}** - Delete system prompt

### 🔧 Test Endpoints
- **GET /test/health** - Health check

## Environment Variables

The collection uses these variables (set them in Postman Environment):

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `base_url` | API server base URL | `http://localhost:8000` |
| `jwt_token` | JWT authentication token | Set after login |
| `profile_id` | Sample profile ID for testing | `550e8400-e29b-41d4-a716-446655440001` |
| `prompt_id` | Sample system prompt ID | `550e8400-e29b-41d4-a716-446655440003` |

## Quick Start Guide

### 1. Set Up Environment
1. Create a new environment in Postman
2. Set `base_url` to your server URL (default: `http://localhost:8000`)
3. Leave `jwt_token` empty initially

### 2. Authenticate
1. **Option A**: Use existing Supabase token
   - Use "Exchange Supabase Token" endpoint
   - Paste your Supabase access token in the request body
   
2. **Option B**: Direct JWT (development only)
   - Use "Create Direct JWT Token" endpoint  
   - Provide a `user_id` in the request body

3. Copy the returned `access_token` from the response
4. Set it as the `jwt_token` environment variable

### 3. Test Existing Endpoints

#### Upload a Document
1. Use "Upload Document" endpoint
2. Attach a PDF file
3. Set `profileID` to a valid UUID (use the sample `profile_id` variable)
4. The document will be processed and embeddings stored

#### Query the RAG System
1. Use "Query RAG Pipeline" endpoint
2. Modify the request body:
   ```json
   {
     "query": "Your question here",
     "profile_id": "{{profile_id}}",
     "k_retrieval": 3,
     "system_prompt": "You are a helpful assistant."
   }
   ```

## Current Implementation Status

### ✅ EXISTING & WORKING
- Authentication (Supabase token exchange)
- Document upload with embedding creation
- RAG query pipeline
- Collection deletion

### 🚧 TO BE IMPLEMENTED
- Profile management endpoints
- System prompts management
- Chat history storage and retrieval
- Enhanced document metadata storage
- JWT middleware for all endpoints

## Request/Response Examples

### Authentication Success Response
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer"
  }
}
```

### Document Upload Success Response
```json
{
  "success": true,
  "data": {
    "document_id": "550e8400-e29b-41d4-a716-446655440002",
    "collection_id": "550e8400-e29b-41d4-a716-446655440001",
    "status": "processing",
    "filename": "document.pdf",
    "detail": "Document uploaded and processing started",
    "chunks_created": 15
  }
}
```

### Chat Query Success Response
```json
{
  "success": true,
  "data": {
    "answer": "Based on the research documents, the main findings include...",
    "source_documents": [
      {
        "page_content": "Relevant content from document...",
        "metadata": {
          "source": "document.pdf",
          "page": 1
        }
      }
    ],
    "profile_id": "550e8400-e29b-41d4-a716-446655440001"
  }
}
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": "Error description here",
  "status_code": 400
}
```

Common HTTP status codes:
- `200` - Success
- `201` - Created successfully  
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

## Notes for Frontend Development

1. **Authentication Flow**: Always include JWT token in `Authorization: Bearer {token}` header
2. **File Upload**: Use `multipart/form-data` for document uploads
3. **Error Handling**: Check the `success` field in responses
4. **Profile Selection**: Users must select a profile before uploading documents or querying
5. **System Prompts**: Will be selectable from a dropdown once implemented

## Testing Tips

1. Start with authentication to get a valid JWT token
2. Test document upload with a small PDF file first
3. Use the uploaded document's collection for chat queries
4. Check the existing endpoints work before implementing new ones
5. Use the provided sample UUIDs for quick testing

## Development Roadmap

1. **Phase 1**: Implement Profile Management endpoints
2. **Phase 2**: Add JWT authentication middleware to all endpoints  
3. **Phase 3**: Implement System Prompts management
4. **Phase 4**: Add Chat History functionality
5. **Phase 5**: Enhanced document metadata and user association 