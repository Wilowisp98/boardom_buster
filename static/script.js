document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('recommendation-form');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading animation
        results.style.display = 'none';
        loading.style.display = 'block';
        
        // Get form data
        const formData = new FormData(form);
        
        // Send request to server
        fetch('/recommend', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Keep loading spinner visible for at least 2 seconds
            const minLoadingTime = 3000; // 3 seconds
            const startTime = new Date().getTime();
            const elapsedTime = new Date().getTime() - startTime;
            const remainingTime = Math.max(0, minLoadingTime - elapsedTime);
            
            setTimeout(() => {
                // Hide loading spinner
                loading.style.display = 'none';
                
                // Clear previous results
                results.innerHTML = '';
                
                if (data.error) {
                    // Display error message
                    const errorTitle = document.createElement('h2');
                    errorTitle.className = 'recommendation-title';
                    errorTitle.textContent = 'Oops! Something Went Wrong';
                    
                    const errorMessage = document.createElement('div');
                    errorMessage.className = 'error-message';
                    errorMessage.textContent = data.error;
                    
                    results.appendChild(errorTitle);
                    results.appendChild(errorMessage);
                } else {
                    // Create header for results
                    const header = document.createElement('h2');
                    header.className = 'recommendation-title';
                    header.textContent = `Games Similar to ${data.input_game}`;
                    results.appendChild(header);
                    
                    // Create list of recommendations
                    data.recommendations.forEach(rec => {
                        const item = document.createElement('div');
                        item.className = 'recommendation-item';
                        
                        const nameElement = document.createElement('div');
                        nameElement.className = 'game-title';
                        nameElement.textContent = rec.name;
                        
                        const explanationElement = document.createElement('div');
                        explanationElement.className = 'why-recommend';
                        explanationElement.textContent = `Why? ${rec.explanation}`;
                        
                        // Add feedback section
                        const feedbackElement = document.createElement('div');
                        feedbackElement.className = 'feedback-section';
                        
                        const feedbackLabel = document.createElement('p');
                        feedbackLabel.className = 'feedback-label';
                        feedbackLabel.textContent = 'Was this recommendation helpful?';
                        
                        const feedbackButtons = document.createElement('div');
                        feedbackButtons.className = 'feedback-buttons';
                        
                        // Create Like button
                        const likeButton = document.createElement('button');
                        likeButton.className = 'feedback-btn like-btn';
                        likeButton.innerHTML = '<span>👍</span> Yes';
                        likeButton.setAttribute('data-game', rec.name);
                        likeButton.setAttribute('data-type', 'like');
                        
                        // Create Dislike button
                        const dislikeButton = document.createElement('button');
                        dislikeButton.className = 'feedback-btn dislike-btn';
                        dislikeButton.innerHTML = '<span>👎</span> No';
                        dislikeButton.setAttribute('data-game', rec.name);
                        dislikeButton.setAttribute('data-type', 'dislike');
                        
                        // Add event listeners for feedback
                        likeButton.addEventListener('click', submitFeedback);
                        dislikeButton.addEventListener('click', submitFeedback);
                        
                        // Append the feedback elements
                        feedbackButtons.appendChild(likeButton);
                        feedbackButtons.appendChild(dislikeButton);
                        feedbackElement.appendChild(feedbackLabel);
                        feedbackElement.appendChild(feedbackButtons);
                        
                        // Append all elements to the item
                        item.appendChild(nameElement);
                        item.appendChild(explanationElement);
                        item.appendChild(feedbackElement);
                        results.appendChild(item);
                    });
                }
                
                // Show results section with a slight delay for animation effect
                setTimeout(() => {
                    results.style.display = 'block';
                }, 200);
            }, remainingTime);
        })
        .catch(error => {
            // Force minimum loading time of 2 seconds
            setTimeout(() => {
                // Hide loading spinner and show error
                loading.style.display = 'none';
                
                // Clear previous results
                results.innerHTML = '';
                
                const errorTitle = document.createElement('h2');
                errorTitle.className = 'recommendation-title';
                errorTitle.textContent = 'Oops! Something Went Wrong';
                
                const errorMessage = document.createElement('div');
                errorMessage.className = 'error-message';
                errorMessage.textContent = 'There was an error connecting to the server. Please try again later.';
                
                results.appendChild(errorTitle);
                results.appendChild(errorMessage);
                results.style.display = 'block';
            }, 2000);
        });
    });
    
    // Function to handle feedback submission
    function submitFeedback(e) {
        e.preventDefault();
        const button = e.currentTarget;
        const gameTitle = button.getAttribute('data-game');
        const feedbackType = button.getAttribute('data-type');
        const inputGame = document.getElementById('game-name').value;
        const parentSection = button.closest('.feedback-section');
        
        if (feedbackType === 'like') {
            // Disable both buttons in the parent container
            const allButtons = parentSection.querySelectorAll('.feedback-btn');
            allButtons.forEach(btn => {
                btn.disabled = true;
                btn.classList.add('disabled');
            });
            
            // Show feedback received message
            const thankYouMsg = document.createElement('p');
            thankYouMsg.className = 'feedback-thanks';
            thankYouMsg.textContent = 'Thanks for your feedback!';
            parentSection.appendChild(thankYouMsg);
            
            // Send feedback to server
            sendFeedbackToServer(inputGame, gameTitle, feedbackType);
        } else if (feedbackType === 'dislike') {
            // Remove the buttons
            parentSection.querySelector('.feedback-buttons').remove();
            parentSection.querySelector('.feedback-label').textContent = 'Why wasn\'t this recommendation helpful?';
            
            // Create detailed feedback form
            const detailedFeedback = document.createElement('div');
            detailedFeedback.className = 'detailed-feedback';
            
            // Add reason options
            const reasonsDiv = document.createElement('div');
            reasonsDiv.className = 'feedback-reasons';
            
            const reasons = [
                'Too similar to games I already know',
                'Too different from what I enjoy',
                'Not my preferred theme/genre',
                'Too complex/simple for my taste',
                'Other reason'
            ];
            
            reasons.forEach(reason => {
                const option = document.createElement('div');
                option.className = 'feedback-reason-option';
                
                const radio = document.createElement('input');
                radio.type = 'radio';
                radio.name = `reason-${gameTitle.replace(/\s+/g, '-')}`;
                radio.id = `reason-${reason.replace(/\s+/g, '-')}-${gameTitle.replace(/\s+/g, '-')}`;
                radio.value = reason;
                
                const label = document.createElement('label');
                label.htmlFor = radio.id;
                label.textContent = reason;
                
                option.appendChild(radio);
                option.appendChild(label);
                reasonsDiv.appendChild(option);
            });
            
            // Add comment area
            const commentDiv = document.createElement('div');
            commentDiv.className = 'feedback-comment';
            
            const commentLabel = document.createElement('label');
            commentLabel.htmlFor = `comment-${gameTitle.replace(/\s+/g, '-')}`;
            commentLabel.textContent = 'Additional comments (optional):';
            
            const commentInput = document.createElement('textarea');
            commentInput.id = `comment-${gameTitle.replace(/\s+/g, '-')}`;
            commentInput.placeholder = 'Tell us more about your thoughts...';
            
            commentDiv.appendChild(commentLabel);
            commentDiv.appendChild(commentInput);
            
            // Add submit button
            const submitBtn = document.createElement('button');
            submitBtn.className = 'feedback-submit';
            submitBtn.textContent = 'Submit Feedback';
            submitBtn.addEventListener('click', function() {
                // Get selected reason
                const selectedReason = reasonsDiv.querySelector('input[type="radio"]:checked');
                let reason = selectedReason ? selectedReason.value : 'No reason provided';
                
                // Get comment
                const comment = commentInput.value.trim();
                
                // Send feedback to server
                sendFeedbackToServer(inputGame, gameTitle, 'dislike', reason, comment);
                
                // Remove the form and show thank you message
                detailedFeedback.innerHTML = '';
                const thankYouMsg = document.createElement('p');
                thankYouMsg.className = 'feedback-thanks';
                thankYouMsg.textContent = 'Thanks for your detailed feedback!';
                detailedFeedback.appendChild(thankYouMsg);
            });
            
            // Assemble the detailed feedback form
            detailedFeedback.appendChild(reasonsDiv);
            detailedFeedback.appendChild(commentDiv);
            detailedFeedback.appendChild(submitBtn);
            
            // Add to parent section
            parentSection.appendChild(detailedFeedback);
        }
    }
    
    // Function to send feedback to server
    function sendFeedbackToServer(inputGame, gameTitle, feedbackType, reason = null, comment = null) {
        const feedbackData = {
            input_game: inputGame,
            recommended_game: gameTitle,
            feedback: feedbackType,
            timestamp: new Date().toISOString()
        };
        
        if (reason) feedbackData.reason = reason;
        if (comment) feedbackData.comment = comment;
        
        fetch('/save_feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(feedbackData)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Feedback saved:', data);
        })
        .catch(error => {
            console.error('Error saving feedback:', error);
        });
    }
    
    // Replace dice numbers with dots
    const diceElements = document.querySelectorAll('.dice');
    diceElements.forEach(dice => {
        // Generate a random number between 1 and 6
        const randomNumber = Math.floor(Math.random() * 6) + 1;
        
        // Clear the text content
        dice.textContent = '';
        
        // Add the appropriate number of dots
        for (let i = 0; i < randomNumber; i++) {
            const dot = document.createElement('div');
            dot.className = 'static-dot';
            dice.appendChild(dot);
        }
        
        // Add a class to help with CSS styling based on the number
        dice.classList.add(`dice-value-${randomNumber}`);
    });
});