css ='''
    <style>
        .chat_message {
            padding:1.5rem; border-radius:0.5rem; margin-bottom:1rem; display: flex
        }
        
        .chat_message.user {
            background-color: #2b313e;
            }
        .chat_message.bot {
            background-color: #475063;
            }
        .chat_message .avatar {
            width: 15%;            
            }        
        .chat_message .avatar img {
            max-width: 78px;
            max-height: 78px;
            border-radius: 50%;
            object-fit: cover;
            }
        .chat_message .message {
            width: 85%;
            padding: 0 1.5rem;
            color: #fff;
            }
    </style>
'''

bot_template = '''
    <div class="chat_message bot">
        <div class="avatar">
            <img src="https://i.ibb.co/qWBwpNb/Photo-logo-5.png" alt="bot avatar">
        </div>
        <div class="message">{{MSG}}</div>
            
    </div>
'''

user_template = '''
    <div class="chat_message user">
        <div class="avatar">
            <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png" alt="user avatar">bot avatar">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
'''