from agent_test import Project1Test
p1 = Project1Test()

p1.test_minimax_interface()
print('Passed Minimax Interface Test')

p1.test_minimax()
print('Passed Minimax Test')

p1.test_alphabeta_interface()
print('Passed Alphabeta Interface Test')

p1.test_alphabeta()
print('Passed Alphabeta Test')

p1.test_get_move()
print('Passed Get Move Test')

p1.test_heuristic()
print('Passed Heuristic Test')
