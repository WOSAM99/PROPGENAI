# Conversation History Fetcher for Supabase Database
# Updated to work with actual Supabase schema and connection

import asyncio
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # type: ignore
from supabase import create_client, Client # type: ignore
import os
from dotenv import load_dotenv # type: ignore   

load_dotenv()

SUPABASE_URL= "https://fqgaitfmuoensaesdwyi.supabase.co"
SUPABASE_ANON_KEY= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZxZ2FpdGZtdW9lbnNhZXNkd3lpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg4NTU4NzAsImV4cCI6MjA2NDQzMTg3MH0.HKVjNvyzohgX3jq4R2VPlWyVyL2vKrTGDDz7BE7zLgQ"


# Debug: Print connection details
print(f"🔧 Using Supabase URL: {SUPABASE_URL}")
print(f"🔧 Using Supabase Key: {SUPABASE_ANON_KEY[:20]}...")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Role mapping for LangChain message types
ROLE_MAP = {
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
    "system": SystemMessage,
}

def fetch_conversation_history(profile_id: str = '850f1278-1a98-4205-aaea-b355353ce75e', limit: int = 6) -> List:
    """
    Fetch conversation history from Supabase messages table for a given profile_id.
    Returns LangChain message objects.
    
    Args:
        profile_id (str): The profile ID to fetch messages for
        limit (int): Number of recent messages to fetch (default: 10)
    
    Returns:
        List: List of LangChain message objects (HumanMessage, AIMessage, etc.)
    """
    try:
        print(f"🔍 Fetching conversation history for profile: {profile_id}")
        
        # Query Supabase messages table
        response = supabase.table('messages') \
            .select('id, role, content, created_at, conversation_id') \
            .eq('profile_id', profile_id) \
            .order('created_at', desc=True) \
            .limit(limit) \
            .execute()
        
        if not response.data:
            print(f"⚠️  No messages found for profile: {profile_id}")
            return []
        
        # Convert to LangChain message objects and reverse to get chronological order
        messages = []
        for row in reversed(response.data):  # Reverse to get oldest first (chronological)
            role = row['role'].lower()
            content = row['content']
            
            # Map role to appropriate LangChain message type
            if role in ROLE_MAP:
                message_class = ROLE_MAP[role]
                messages.append(message_class(content=content))
            else:
                # Default to HumanMessage if role is unknown
                print(f"⚠️  Warning: Unknown role '{role}', defaulting to HumanMessage")
                messages.append(HumanMessage(content=content))
        
        print(f"✅ Successfully fetched {len(messages)} messages for profile {profile_id}")
        return messages
        
    except Exception as e:
        print(f"❌ Error fetching conversation history: {e}")
        return []

def fetch_conversation_as_context_string(profile_id: str = '850f1278-1a98-4205-aaea-b355353ce75e', limit: int = 10) -> str:
    """
    Fetch conversation history and return as a formatted context string.
    This is perfect for RAG system integration.
    
    Args:
        profile_id (str): The profile ID to fetch messages for
        limit (int): Number of recent messages to fetch (default: 10)
    
    Returns:
        str: Formatted conversation history as a string
    """
    try:
        print(f"🔍 Fetching conversation context for profile: {profile_id}")
        
        # Query Supabase messages table
        response = supabase.table('messages') \
            .select('role, content, created_at') \
            .eq('profile_id', profile_id) \
            .order('created_at', desc=True) \
            .limit(limit) \
            .execute()
        
        if not response.data:
            print(f"⚠️  No conversation history found for profile: {profile_id}")
            return "No previous conversation history available."
        
        # Format as context string (reverse to get chronological order)
        conversation_lines = []
        for row in reversed(response.data):
            role = row['role'].upper()
            content = row['content']
            conversation_lines.append(f"{role}: {content}")
        
        context = "\n".join(conversation_lines)
        print(f"✅ Formatted conversation context ({len(response.data)} messages)")
        return context
        
    except Exception as e:
        print(f"❌ Error fetching conversation context: {e}")
        return "Error retrieving conversation history."

def fetch_conversation_as_dict_list(profile_id: str = '850f1278-1a98-4205-aaea-b355353ce75e', limit: int = 10) -> List[Dict]:
    """
    Fetch conversation history as a list of dictionaries.
    Useful for JSON serialization and API responses.
    
    Args:
        profile_id (str): The profile ID to fetch messages for
        limit (int): Number of recent messages to fetch (default: 10)
    
    Returns:
        List[Dict]: List of message dictionaries with full metadata
    """
    try:
        print(f"🔍 Fetching conversation as dict list for profile: {profile_id}")
        
        # Query Supabase messages table with all fields
        response = supabase.table('messages') \
            .select('id, role, content, created_at, conversation_id, system_prompt_id, metadata') \
            .eq('profile_id', profile_id) \
            .order('created_at', desc=True) \
            .limit(limit) \
            .execute()
        
        if not response.data:
            print(f"⚠️  No messages found for profile: {profile_id}")
            return []
        
        # Convert to list of dictionaries in chronological order
        messages = []
        for row in reversed(response.data):
            messages.append({
                'id': row['id'],
                'role': row['role'],
                'content': row['content'],
                'created_at': row['created_at'],
                'conversation_id': row['conversation_id'],
                'system_prompt_id': row['system_prompt_id'],
                'metadata': row['metadata']
            })
        
        print(f"✅ Fetched {len(messages)} messages as dict list")
        return messages
        
    except Exception as e:
        print(f"❌ Error fetching conversation as dict list: {e}")
        return []

def get_conversation_stats(profile_id: str = '850f1278-1a98-4205-aaea-b355353ce75e') -> Dict:
    """
    Get conversation statistics for a profile.
    
    Args:
        profile_id (str): The profile ID to get stats for
    
    Returns:
        Dict: Statistics about conversations and messages
    """
    try:
        print(f"📊 Getting conversation stats for profile: {profile_id}")
        
        # Get total message count
        messages_response = supabase.table('messages') \
            .select('id', count='exact') \
            .eq('profile_id', profile_id) \
            .execute()
        
        # Get conversation count
        conversations_response = supabase.table('conversations') \
            .select('id', count='exact') \
            .eq('profile_id', profile_id) \
            .execute()
        
        # Get role distribution
        role_response = supabase.table('messages') \
            .select('role') \
            .eq('profile_id', profile_id) \
            .execute()
        
        role_counts = {}
        if role_response.data:
            for msg in role_response.data:
                role = msg['role']
                role_counts[role] = role_counts.get(role, 0) + 1
        
        stats = {
            'profile_id': profile_id,
            'total_messages': messages_response.count or 0,
            'total_conversations': conversations_response.count or 0,
            'role_distribution': role_counts,
            'has_conversations': (messages_response.count or 0) > 0
        }
        
        print(f"✅ Stats: {stats['total_messages']} messages, {stats['total_conversations']} conversations")
        return stats
        
    except Exception as e:
        print(f"❌ Error getting conversation stats: {e}")
        return {
            'profile_id': profile_id,
            'total_messages': 0,
            'total_conversations': 0,
            'role_distribution': {},
            'has_conversations': False,
            'error': str(e)
        }

def view_conversation_history(profile_id: str = '850f1278-1a98-4205-aaea-b355353ce75e', limit: int = 10):
    """
    Simple function to view conversation history in a readable format.
    Perfect for debugging and seeing what data is actually fetched.
    """
    print(f"\n🔍 VIEWING CONVERSATION HISTORY")
    print(f"Profile ID: {profile_id}")
    print(f"Limit: {limit}")
    print("=" * 80)
    
    try:
        # Get stats first
        stats = get_conversation_stats(profile_id)
        print(f"📊 PROFILE STATS:")
        print(f"   • Total Messages: {stats['total_messages']}")
        print(f"   • Total Conversations: {stats['total_conversations']}")
        print(f"   • Role Distribution: {stats['role_distribution']}")
        print(f"   • Has Conversations: {stats['has_conversations']}")
        
        if stats['total_messages'] == 0:
            print("\n⚠️  No messages found for this profile!")
            
            # Try to find any profile with messages
            print("🔍 Looking for any profile with messages...")
            try:
                any_messages = supabase.table('messages').select('profile_id').limit(3).execute()
                if any_messages.data:
                    available_profiles = list(set([msg['profile_id'] for msg in any_messages.data if msg['profile_id']]))
                    print(f"📋 Found profiles with messages: {available_profiles[:2]}")
                    if available_profiles:
                        profile_id = available_profiles[0]
                        print(f"🔄 Switching to profile: {profile_id}")
                        stats = get_conversation_stats(profile_id)
                else:
                    print("❌ No messages found in the entire database!")
                    return
            except Exception as e:
                print(f"❌ Error finding profiles: {e}")
                return
        
        print(f"\n📝 CONVERSATION HISTORY (Last {limit} messages):")
        print("-" * 80)
        
        # Get the actual conversation data
        response = supabase.table('messages') \
            .select('id, role, content, created_at, conversation_id') \
            .eq('profile_id', profile_id) \
            .order('created_at', desc=True) \
            .limit(limit) \
            .execute()
        
        if not response.data:
            print("⚠️  No conversation history found!")
            return
        
        # Display messages in chronological order (oldest first)
        for i, msg in enumerate(reversed(response.data), 1):
            role_emoji = "👤" if msg['role'].lower() == 'user' else "🤖"
            role = msg['role'].upper()
            content = msg['content']
            created_at = msg['created_at']
            
            print(f"\n{i}. {role_emoji} {role} [{created_at}]")
            print(f"   ID: {msg['id'][:8]}...")
            print(f"   Content: {content}")
            print(f"   Conversation: {msg['conversation_id'][:8] if msg['conversation_id'] else 'None'}...")
        
        print(f"\n✅ Successfully displayed {len(response.data)} messages")
        
        # Show context string format
        context = fetch_conversation_as_context_string(profile_id, limit)
        print(f"\n📋 AS CONTEXT STRING (for RAG):")
        print("-" * 50)
        print(context)
        
    except Exception as e:
        print(f"❌ Error viewing conversation history: {e}")

def test_conversation_functions():
    """
    Test function to verify all conversation fetching functions work with real Supabase data.
    """
    print("🧪 Testing Conversation Functions with Supabase")
    print("=" * 60)
    
    test_profile_id = '850f1278-1a98-4205-aaea-b355353ce75e'
    
    # Test 1: Get conversation stats
    print("\n1. Testing get_conversation_stats():")
    print("-" * 40)
    stats = get_conversation_stats(test_profile_id)
    print(f"📊 Profile Stats:")
    print(f"   Total Messages: {stats['total_messages']}")
    print(f"   Total Conversations: {stats['total_conversations']}")
    print(f"   Role Distribution: {stats['role_distribution']}")
    print(f"   Has Conversations: {stats['has_conversations']}")
    
    if stats['total_messages'] == 0:
        print("⚠️  No messages found for this profile. Testing with any available profile...")
        
        # Try to find any profile with messages
        try:
            all_messages = supabase.table('messages').select('profile_id').limit(1).execute()
            if all_messages.data:
                test_profile_id = all_messages.data[0]['profile_id']
                print(f"🔄 Using profile: {test_profile_id}")
                stats = get_conversation_stats(test_profile_id)
                print(f"📊 New Profile Stats: {stats}")
        except Exception as e:
            print(f"❌ Could not find any messages: {e}")
            return
    
    # Test 2: Fetch as LangChain objects
    print("\n2. Testing fetch_conversation_history() - LangChain Objects:")
    print("-" * 40)
    messages = fetch_conversation_history(test_profile_id, limit=5)
    for i, msg in enumerate(messages):
        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"  Message {i+1}: {type(msg).__name__}")
        print(f"  Content: {content_preview}")
        print()
    
    # Test 3: Fetch as context string
    print("\n3. Testing fetch_conversation_as_context_string() - String Format:")
    print("-" * 40)
    context = fetch_conversation_as_context_string(test_profile_id, limit=5)
    print("📝 Context String:")
    print(context)
    
    # Test 4: Fetch as dictionary list
    print("\n4. Testing fetch_conversation_as_dict_list() - Dictionary Format:")
    print("-" * 40)
    dict_messages = fetch_conversation_as_dict_list(test_profile_id, limit=3)
    for i, msg in enumerate(dict_messages):
        content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  Message {i+1}: {msg['role']} - {content_preview}")
        print(f"    ID: {msg['id']}")
        print(f"    Created: {msg['created_at']}")
        print()
    
    print("✅ All tests completed!")

def test_with_different_profiles():
    """
    Test with multiple profile IDs to see the difference.
    """
    print("\n🔄 Testing with Multiple Profiles:")
    print("-" * 50)
    
    # Get a few different profile IDs from the database
    try:
        profiles_response = supabase.table('messages') \
            .select('profile_id') \
            .limit(5) \
            .execute()
        
        if profiles_response.data:
            unique_profiles = list(set([msg['profile_id'] for msg in profiles_response.data if msg['profile_id']]))
            
            for profile_id in unique_profiles[:3]:  # Test up to 3 profiles
                print(f"\n📋 Profile: {profile_id}")
                stats = get_conversation_stats(profile_id)
                print(f"   Messages: {stats['total_messages']}, Conversations: {stats['total_conversations']}")
                
                if stats['total_messages'] > 0:
                    context = fetch_conversation_as_context_string(profile_id, limit=2)
                    preview = context[:200] + "..." if len(context) > 200 else context
                    print(f"   Latest Context Preview: {preview}")
        else:
            print("⚠️  No profiles found with messages")
            
    except Exception as e:
        print(f"❌ Error testing multiple profiles: {e}")

# Main execution for testing
if __name__ == "__main__":
    print("🚀 Starting Supabase Conversation History Testing")
    print("=" * 70)
    
    try:
        # Test database connection
        print("🔌 Testing Supabase connection...")
        test_response = supabase.table('messages').select('id').limit(1).execute()
        print("✅ Supabase connection successful!")
        
        # First, show a simple view of the conversation history
        view_conversation_history()
        
        # Run all tests
        test_conversation_functions()
        test_with_different_profiles()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("🔧 Check your SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
    
    print("\n" + "=" * 70)
    print("📝 Usage Notes:")
    print("1. ✅ Using real Supabase database connection")
    print("2. ✅ Fetching from actual 'messages' table")
    print("3. ✅ Profile ID format: UUID string")
    print("4. ✅ Supports role mapping: user/assistant → HumanMessage/AIMessage")
    print("5. ✅ Returns empty lists/strings if no messages found")
    print("6. 🎯 Best function for RAG: fetch_conversation_as_context_string()")

# ===== SIMPLE EXECUTION ALTERNATIVES =====

def just_view_history(profile_id: str = '850f1278-1a98-4205-aaea-b355353ce75e'):
    """
    Just view conversation history without running all tests.
    Usage: python -c "from app.RAG.conv import just_view_history; just_view_history()"
    """
    try:
        print("🔌 Testing Supabase connection...")
        test_response = supabase.table('messages').select('id').limit(1).execute()
        print("✅ Connection successful!")
        
        view_conversation_history(profile_id)
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Uncomment one of these lines to change what runs when you execute the file:

# To just view history for a specific profile:
# if __name__ == "__main__": just_view_history('your-profile-id-here')

# To run full test suite (current default):
# if __name__ == "__main__": [full test code below]
