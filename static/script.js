// Global state
let currentUserType = null;
let chatInitialized = false;
let chatHistory = [];

// API Base URL
const API_BASE = '';

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

function setupEventListeners() {
    // User type selection
    document.querySelectorAll('.user-card').forEach(card => {
        card.addEventListener('click', function() {
            selectUserType(this.dataset.type);
        });
    });

    // Initialize button
    document.getElementById('initBtn').addEventListener('click', initializeChatbot);

    // Send message
    document.getElementById('sendBtn').addEventListener('click', sendMessage);
    document.getElementById('chatInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Generate timetable
    document.getElementById('generateTimetableBtn').addEventListener('click', generateTimetable);

    // Reset button
    document.getElementById('resetBtn').addEventListener('click', resetApp);
}

function selectUserType(userType) {
    currentUserType = userType;
    document.getElementById('userSelection').style.display = 'none';
    document.getElementById('mainContent').style.display = 'grid';
    
    // Update UI based on user type
    const userBadge = document.getElementById('userBadge');
    const chatTitle = document.getElementById('chatTitle');
    const timetableSection = document.getElementById('timetableSection');
    
    const userTypes = {
        'student': { name: 'Student', icon: '👨‍🎓' },
        'parent': { name: 'Parent', icon: '👨‍👩‍👧' },
        'faculty': { name: 'Faculty', icon: '👨‍🏫' }
    };
    
    const userInfo = userTypes[userType];
    userBadge.textContent = `${userInfo.icon} ${userInfo.name}`;
    chatTitle.textContent = `Chat with VCET CSE Assistant (${userInfo.name})`;
    
    // Show timetable section only for faculty
    if (userType === 'faculty') {
        timetableSection.style.display = 'block';
    } else {
        timetableSection.style.display = 'none';
    }
    
    // Check if chatbot is initialized on backend
    checkInitializationStatus();
    
    // Clear chat
    clearChat();
    addWelcomeMessage();
}

function checkInitializationStatus() {
    // Check backend initialization status
    fetch('/api/check-initialized')
        .then(response => response.json())
        .then(data => {
            if (data.initialized) {
                // Chatbot is initialized, enable chat
                chatInitialized = true;
                document.getElementById('chatInput').disabled = false;
                document.getElementById('sendBtn').disabled = false;
                document.getElementById('initBtn').innerHTML = '<i class="fas fa-check"></i> Initialized';
                document.getElementById('initBtn').style.background = 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)';
                document.getElementById('initBtn').disabled = true;
                document.getElementById('initStatus').className = 'status success';
                document.getElementById('initStatus').textContent = '✓ Chatbot is initialized and ready';
                document.getElementById('initStatus').style.display = 'block';
                
                // Update welcome message
                document.getElementById('chatMessages').innerHTML = `
                    <div class="message assistant">
                        <div class="message-header">Assistant</div>
                        <div>Hello! I'm your VCET CSE Department assistant. How can I help you today?</div>
                    </div>
                `;
            } else {
                // Chatbot not initialized, show initialize button
                chatInitialized = false;
                document.getElementById('chatInput').disabled = true;
                document.getElementById('sendBtn').disabled = true;
                document.getElementById('initBtn').innerHTML = '<i class="fas fa-power-off"></i> Initialize Chatbot';
                document.getElementById('initBtn').style.background = '';
                document.getElementById('initBtn').disabled = false;
                document.getElementById('initStatus').style.display = 'none';
            }
        })
        .catch(error => {
            // On error, assume not initialized
            chatInitialized = false;
            document.getElementById('chatInput').disabled = true;
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('initBtn').innerHTML = '<i class="fas fa-power-off"></i> Initialize Chatbot';
            document.getElementById('initBtn').style.background = '';
            document.getElementById('initBtn').disabled = false;
        });
}

function addWelcomeMessage() {
    const messagesDiv = document.getElementById('chatMessages');
    messagesDiv.innerHTML = `
        <div class="welcome-message">
            <i class="fas fa-robot"></i>
            <p>Welcome! Please initialize the chatbot to start chatting.</p>
        </div>
    `;
}

function clearChat() {
    chatHistory = [];
    document.getElementById('chatMessages').innerHTML = '';
}

function initializeChatbot() {
    const initBtn = document.getElementById('initBtn');
    const statusDiv = document.getElementById('initStatus');
    
    initBtn.disabled = true;
    initBtn.innerHTML = '<span class="loading"></span> Initializing...';
    statusDiv.className = 'status info';
    statusDiv.textContent = 'Initializing RAG system...';
    statusDiv.style.display = 'block';
    
    fetch('/api/initialize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            chatInitialized = true;
            statusDiv.className = 'status success';
            statusDiv.textContent = '✓ ' + data.message;
            initBtn.innerHTML = '<i class="fas fa-check"></i> Initialized';
            initBtn.style.background = 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)';
            
            // Enable chat
            document.getElementById('chatInput').disabled = false;
            document.getElementById('sendBtn').disabled = false;
            
            // Update welcome message
            document.getElementById('chatMessages').innerHTML = `
                <div class="message assistant">
                    <div class="message-header">Assistant</div>
                    <div>Hello! I'm your VCET CSE Department assistant. How can I help you today?</div>
                </div>
            `;
        } else {
            statusDiv.className = 'status error';
            statusDiv.textContent = '✗ ' + data.message;
            initBtn.disabled = false;
            initBtn.innerHTML = '<i class="fas fa-power-off"></i> Initialize Chatbot';
            chatInitialized = false;
        }
    })
    .catch(error => {
        statusDiv.className = 'status error';
        let errorMsg = 'Error: ';
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            errorMsg += 'Cannot connect to server. Please make sure the Flask server is running on http://localhost:5000';
        } else {
            errorMsg += error.message;
        }
        statusDiv.textContent = '✗ ' + errorMsg;
        initBtn.disabled = false;
        initBtn.innerHTML = '<i class="fas fa-power-off"></i> Initialize Chatbot';
        chatInitialized = false;
    });
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message || !chatInitialized) return;
    
    // Add user message to chat
    addMessage('user', message);
    input.value = '';
    
    // Disable input while processing
    input.disabled = true;
    document.getElementById('sendBtn').disabled = true;
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    // Send to API
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            question: message,
            user_type: currentUserType
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        removeTypingIndicator(typingId);
        
        if (data.success) {
            addMessage('assistant', data.answer);
        } else {
            addMessage('assistant', 'Sorry, I encountered an error: ' + data.message);
            // If initialization error, reset chat initialized flag
            if (data.message.includes('initialize') || data.message.includes('Vectorstore')) {
                chatInitialized = false;
                document.getElementById('initBtn').innerHTML = '<i class="fas fa-power-off"></i> Initialize Chatbot';
                document.getElementById('initBtn').style.background = '';
            }
        }
        
        // Re-enable input
        input.disabled = false;
        document.getElementById('sendBtn').disabled = false;
        input.focus();
    })
    .catch(error => {
        removeTypingIndicator(typingId);
        let errorMsg = 'Sorry, I encountered an error: ';
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            errorMsg += 'Cannot connect to server. Please make sure the Flask server is running on http://localhost:5000';
        } else {
            errorMsg += error.message;
        }
        addMessage('assistant', errorMsg);
        input.disabled = false;
        document.getElementById('sendBtn').disabled = false;
    });
}

function addMessage(role, content) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const header = role === 'user' ? 'You' : 'Assistant';
    messageDiv.innerHTML = `
        <div class="message-header">${header}</div>
        <div>${formatMessage(content)}</div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    chatHistory.push({ role, content });
}

function addTypingIndicator() {
    const messagesDiv = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'message assistant';
    typingDiv.innerHTML = `
        <div class="message-header">Assistant</div>
        <div><span class="loading"></span> Thinking...</div>
    `;
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return 'typing-indicator';
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

function formatMessage(content) {
    if (!content) return '';
    
    // Remove escaped characters first
    content = content.replace(/\\\*/g, '*');
    content = content.replace(/\\#/g, '#');
    content = content.replace(/\\n/g, '\n');
    
    // Convert markdown headings
    content = content.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    content = content.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    content = content.replace(/^# (.*$)/gim, '<h2>$1</h2>');
    
    // Convert markdown bold (handle multiple formats)
    content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert markdown code blocks
    content = content.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    content = content.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Convert bullet lists (handle various formats)
    // First, split by lines to process lists properly
    let lines = content.split('\n');
    let inList = false;
    let formattedLines = [];
    let listItems = [];
    
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trim();
        
        // Detect bullet points (various formats)
        if (line.match(/^[\*\-\+]\s+(.+)$/) || line.match(/^\d+[\.\)]\s+(.+)$/)) {
            if (!inList) {
                inList = true;
                listItems = [];
            }
            // Extract the content after bullet
            let match = line.match(/^[\*\-\+\d\.\)]+\s+(.+)$/);
            if (match) {
                listItems.push(match[1]);
            }
        } else {
            // End of list
            if (inList && listItems.length > 0) {
                formattedLines.push('<ul>' + listItems.map(item => `<li>${formatInlineMarkdown(item)}</li>`).join('') + '</ul>');
                listItems = [];
                inList = false;
            }
            
            // Process non-list lines
            if (line.length > 0) {
                formattedLines.push(formatInlineMarkdown(line));
            } else {
                formattedLines.push('<br>');
            }
        }
    }
    
    // Handle remaining list items
    if (inList && listItems.length > 0) {
        formattedLines.push('<ul>' + listItems.map(item => `<li>${formatInlineMarkdown(item)}</li>`).join('') + '</ul>');
    }
    
    // Join lines
    content = formattedLines.join('\n');
    
    // Convert remaining newlines to <br>
    content = content.replace(/\n/g, '<br>');
    
    // Clean up multiple consecutive <br> tags (max 2)
    content = content.replace(/(<br>\s*){3,}/g, '<br><br>');
    
    // Remove trailing asterisks and formatting artifacts
    content = content.replace(/\*+$/, '');
    content = content.replace(/^[\*\#\s]+$/, '');
    
    return content;
}

function formatInlineMarkdown(text) {
    if (!text) return '';
    
    // Convert bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert inline code
    text = text.replace(/`(.*?)`/g, '<code>$1</code>');
    
    return text;
}

function generateTimetable() {
    const facultyName = document.getElementById('facultyName').value.trim();
    const semester1 = document.getElementById('semester1').value;
    const semester2 = document.getElementById('semester2').value;
    const statusDiv = document.getElementById('timetableStatus');
    const btn = document.getElementById('generateTimetableBtn');
    
    if (!facultyName) {
        statusDiv.className = 'status error';
        statusDiv.textContent = 'Please enter faculty name';
        statusDiv.style.display = 'block';
        return;
    }
    
    if (semester1 === semester2) {
        statusDiv.className = 'status error';
        statusDiv.textContent = 'Please select two different semesters';
        statusDiv.style.display = 'block';
        return;
    }
    
    if (!chatInitialized) {
        statusDiv.className = 'status error';
        statusDiv.textContent = 'Please initialize the chatbot first';
        statusDiv.style.display = 'block';
        return;
    }
    
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Generating...';
    statusDiv.className = 'status info';
    statusDiv.textContent = 'Generating timetable... This may take a moment.';
    statusDiv.style.display = 'block';
    
    fetch('/api/generate-timetable', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            faculty_name: facultyName,
            semester1: semester1,
            semester2: semester2
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            statusDiv.className = 'status success';
            statusDiv.textContent = '✓ ' + data.message;
            
            // Create download link
            const downloadLink = document.createElement('a');
            downloadLink.href = `/api/download/${data.file_path}`;
            downloadLink.download = data.file_path;
            downloadLink.className = 'btn btn-success';
            downloadLink.style.marginTop = '10px';
            downloadLink.innerHTML = '<i class="fas fa-download"></i> Download PDF';
            downloadLink.click();
            
            // Show subject info
            addMessage('assistant', `Timetable generated successfully!\n\nSelected Subjects:\n• ${data.subject1} (${semester1})\n• ${data.subject2} (${semester2})\n\nClick the download button to get your PDF.`);
        } else {
            statusDiv.className = 'status error';
            statusDiv.textContent = '✗ ' + data.message;
        }
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-file-pdf"></i> Generate Timetable';
    })
    .catch(error => {
        statusDiv.className = 'status error';
        statusDiv.textContent = '✗ Error: ' + error.message;
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-file-pdf"></i> Generate Timetable';
    });
}

function resetApp() {
    if (confirm('Are you sure you want to reset? This will clear all chat history and return to user selection.')) {
        // Reset frontend state
        currentUserType = null;
        chatInitialized = false;
        chatHistory = [];
        
        // Reset UI
        document.getElementById('userSelection').style.display = 'block';
        document.getElementById('mainContent').style.display = 'none';
        document.getElementById('initBtn').innerHTML = '<i class="fas fa-power-off"></i> Initialize Chatbot';
        document.getElementById('initBtn').style.background = '';
        document.getElementById('initBtn').disabled = false;
        document.getElementById('initStatus').style.display = 'none';
        document.getElementById('chatInput').disabled = true;
        document.getElementById('sendBtn').disabled = true;
        
        // Note: We don't reset backend state here because chatbot should remain initialized
        // If user wants to fully reset, they can restart the Flask server
    }
}


