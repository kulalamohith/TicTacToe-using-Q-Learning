import streamlit as st
import numpy as np
from TicTacToe import TicTacToe
from PlayerSQN import PlayerSQN
import time

# Page config
st.set_page_config(
    page_title="TicTacToe RL - AI Algorithm Demo",
    page_icon="🎮",
    layout="wide"
)

# Initialize session state
if 'board' not in st.session_state:
    st.session_state.board = [0] * 9
if 'current_player' not in st.session_state:
    st.session_state.current_player = 1  # 1 = Human (X), 2 = AI (O)
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'winner' not in st.session_state:
    st.session_state.winner = None
if 'stats' not in st.session_state:
    st.session_state.stats = {'ai_wins': 0, 'human_wins': 0, 'draws': 0}
if 'ai_player' not in st.session_state:
    try:
        st.session_state.ai_player = PlayerSQN(epsilon=0.01)
    except:
        st.session_state.ai_player = None

def check_winner(board):
    """Check if there's a winner"""
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    for line in lines:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]
    return None

def is_board_full(board):
    """Check if board is full"""
    return all(cell != 0 for cell in board)

def make_move(position, player):
    """Make a move on the board"""
    if st.session_state.board[position] == 0 and not st.session_state.game_over:
        st.session_state.board[position] = player
        return True
    return False

def ai_move():
    """Get AI move"""
    if st.session_state.ai_player:
        try:
            move = st.session_state.ai_player.move(st.session_state.board.copy())
            return move
        except Exception as e:
            st.error(f"AI Error: {e}")
            return None
    return None

def reset_game():
    """Reset the game"""
    st.session_state.board = [0] * 9
    st.session_state.current_player = 1
    st.session_state.game_over = False
    st.session_state.winner = None

def handle_click(position):
    """Handle cell click"""
    if st.session_state.current_player == 1 and not st.session_state.game_over:
        if make_move(position, 1):
            # Check for human win
            if check_winner(st.session_state.board) == 1:
                st.session_state.winner = 1
                st.session_state.game_over = True
                st.session_state.stats['human_wins'] += 1
                return
            
            # Check for draw
            if is_board_full(st.session_state.board):
                st.session_state.winner = 0
                st.session_state.game_over = True
                st.session_state.stats['draws'] += 1
                return
            
            # AI turn
            st.session_state.current_player = 2
            st.rerun()

def render_board():
    board = st.session_state.board
    for row in range(3):
        cols = st.columns(3, gap="small")
        for col in range(3):
            idx = row * 3 + col
            cell_value = board[idx]
            style = "font-size:2.5rem; height:60px; width:60px; text-align:center; border:2px solid #888; border-radius:8px; background:#fff; display:flex; align-items:center; justify-content:center; margin:auto;"
            if cell_value == 0 and not st.session_state.game_over and st.session_state.current_player == 1:
                if cols[col].button(" ", key=f"cell_{idx}", help=f" {row+1},{col+1}", use_container_width=True):
                    handle_click(idx)
            elif cell_value == 1:
                cols[col].markdown(f"<div style='{style} color:#e74c3c;'>❌</div>", unsafe_allow_html=True)
            elif cell_value == 2:
                cols[col].markdown(f"<div style='{style} color:#3498db;'>⭕</div>", unsafe_allow_html=True)
            else:
                cols[col].markdown(f"<div style='{style}'></div>", unsafe_allow_html=True)

# Main app
st.title("🎮 TicTacToe Reinforcement Learning")
st.markdown("### Challenge the AI trained with Q-Learning & Neural Networks")

# Sidebar with AI info
with st.sidebar:
    st.header(" AI Algorithm Details")
    st.markdown("""
    **Q-Learning Components:**
    - Neural Network: 2 hidden layers (64 neurons each)
    - Training: 5000 episodes with experience replay
    - Strategy: Epsilon-greedy exploration (1.0 → 0.1)
    - Reward System: +1 win, -1 loss, 0 draw
    
    **Model Performance:**
    - Win Rate: ~95% after training
    - Training Time: ~5-10 minutes
    - Inference Speed: <100ms per move
    """)
    
    st.header(" Game Statistics")
    st.metric("AI Wins", st.session_state.stats['ai_wins'])
    st.metric("Your Wins", st.session_state.stats['human_wins'])
    st.metric("Draws", st.session_state.stats['draws'])
    
    if st.button(" New Game"):
        reset_game()
        st.rerun()

# Main game area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Game status
    if st.session_state.game_over:
        if st.session_state.winner == 1:
            st.success(" You win!")
        elif st.session_state.winner == 2:
            st.error(" AI wins!")
        else:
            st.info(" It's a draw!")
    elif st.session_state.current_player == 1:
        st.info("Your turn (❌)")
    else:
        st.warning("AI is thinking...")
    
    # Game board
    st.markdown("---")
    render_board()
    st.markdown("---")

# AI move logic
if st.session_state.current_player == 2 and not st.session_state.game_over:
    with st.spinner("AI is thinking..."):
        time.sleep(0.5)  # Small delay for better UX
        ai_move_pos = ai_move()
        
        if ai_move_pos is not None and ai_move_pos >= 0 and ai_move_pos < 9:
            if make_move(ai_move_pos, 2):
                # Check for AI win
                if check_winner(st.session_state.board) == 2:
                    st.session_state.winner = 2
                    st.session_state.game_over = True
                    st.session_state.stats['ai_wins'] += 1
                
                # Check for draw
                elif is_board_full(st.session_state.board):
                    st.session_state.winner = 0
                    st.session_state.game_over = True
                    st.session_state.stats['draws'] += 1
                
                # Human turn
                else:
                    st.session_state.current_player = 1
                
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with  using Reinforcement Learning and Streamlit</p>
    <p><strong>Technologies:</strong> Python, TensorFlow/Keras, Q-Learning, Neural Networks</p>
</div>
""", unsafe_allow_html=True) 