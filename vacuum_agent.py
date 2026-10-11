"""Simple Reactive Agent for a vacuum cleaner in a 2x2 grid.

Implements the condition-action rule-based reactive architecture with four
explicit components: rule set, interpret_input, rule_match, and agent_program
(execute_action performs the selected action on the environment).
"""

from typing import Callable, Dict, List, Optional, Tuple

# Type aliases for clarity
Position = Tuple[int, int]
State = Dict[str, object]
Rule = Tuple[str, Callable[[State], bool]]

GRID_SIZE = 2
MAX_STEPS = 20

DIRTY = "DIRTY"
CLEAN = "CLEAN"

SUCK = "SUCK"
MOVE_RIGHT = "MOVE_RIGHT"
MOVE_LEFT = "MOVE_LEFT"
MOVE_UP = "MOVE_UP"
MOVE_DOWN = "MOVE_DOWN"


# --------------------------------------------------------------------------
# Rule Set: predefined collection of condition-action rules
# --------------------------------------------------------------------------
def _rule_suck(state: State) -> bool:
    """Condition: current cell is DIRTY."""
    return state["status"] == DIRTY


def _make_move_rule(dr: int, dc: int) -> Callable[[State], bool]:
    """Build the condition for a directional move rule.

    Condition: current cell is CLEAN, the neighbor in the given direction
    exists inside the grid, and that neighbor has not been visited yet.
    """

    def condition(state: State) -> bool:
        if state["status"] != CLEAN:
            return False
        row, col = state["position"]  # type: ignore[misc]
        nrow, ncol = row + dr, col + dc
        if not (0 <= nrow < GRID_SIZE and 0 <= ncol < GRID_SIZE):
            return False
        return (nrow, ncol) not in state["visited"]

    return condition


def build_rule_set() -> List[Rule]:
    """Return the agent's predefined condition-action rule set.

    Rules are ordered by priority: the first rule whose condition matches is
    selected, so cleaning (SUCK) always takes precedence over movement, and
    movement directions are tried in a fixed order.
    """
    return [
        ("IF current cell is DIRTY -> SUCK", _rule_suck),
        (
            "IF current cell is CLEAN AND right neighbor exists AND not "
            "visited -> MOVE_RIGHT",
            _make_move_rule(0, 1),
        ),
        (
            "IF current cell is CLEAN AND left neighbor exists AND not "
            "visited -> MOVE_LEFT",
            _make_move_rule(0, -1),
        ),
        (
            "IF current cell is CLEAN AND upper neighbor exists AND not "
            "visited -> MOVE_UP",
            _make_move_rule(-1, 0),
        ),
        (
            "IF current cell is CLEAN AND lower neighbor exists AND not "
            "visited -> MOVE_DOWN",
            _make_move_rule(1, 0),
        ),
    ]


# Map a rule description to the action its "then" clause prescribes.
def rule_to_action(rule: Rule) -> str:
    """Extract the action (the 'then' part) from a matched condition-action rule."""
    description = rule[0]
    if description.endswith("SUCK"):
        return SUCK
    if description.endswith("MOVE_RIGHT"):
        return MOVE_RIGHT
    if description.endswith("MOVE_LEFT"):
        return MOVE_LEFT
    if description.endswith("MOVE_UP"):
        return MOVE_UP
    if description.endswith("MOVE_DOWN"):
        return MOVE_DOWN
    raise ValueError(f"Unknown rule: {description}")


# --------------------------------------------------------------------------
# interpret_input: percept -> state
# --------------------------------------------------------------------------
def interpret_input(percept: Dict[str, object]) -> State:
    """Convert a raw percept into the agent's internal environment state.

    The percept is the agent's local observation of the world (current
    position + cell status). This function formats it as the state dictionary
    consumed by rule_match, including the set of visited cells so rules can
    test 'not visited'.
    """
    return {
        "position": percept["position"],
        "status": percept["status"],
        "visited": set(percept.get("visited", set())),
    }


# --------------------------------------------------------------------------
# rule_match: (rules, state) -> rule
# --------------------------------------------------------------------------
def rule_match(rules: List[Rule], state: State) -> Optional[Rule]:
    """Select the first applicable rule for the current state.

    Scans the rule set in priority order and returns the first rule whose
    condition evaluates to True; returns None if no rule applies (the agent
    then does nothing for that step).
    """
    for rule in rules:
        if rule[1](state):
            return rule
    return None


# --------------------------------------------------------------------------
# execute_action: apply the chosen action to the environment
# --------------------------------------------------------------------------
def execute_action(action: str, position: Position,
                   grid: List[List[str]]) -> Tuple[Position, bool]:
    """Apply the agent's chosen action to the environment.

    SUCK cleans the current cell; MOVE_* attempts to shift the agent one cell
    while respecting grid boundaries (moves outside the grid are rejected and
    the position stays unchanged). Returns the new position and whether the
    action was a successful movement.
    """
    row, col = position

    if action == SUCK:
        grid[row][col] = CLEAN
        return position, False

    deltas = {
        MOVE_RIGHT: (0, 1),
        MOVE_LEFT: (0, -1),
        MOVE_UP: (-1, 0),
        MOVE_DOWN: (1, 0),
    }
    if action in deltas:
        dr, dc = deltas[action]
        nrow, ncol = row + dr, col + dc
        if 0 <= nrow < GRID_SIZE and 0 <= ncol < GRID_SIZE:
            return (nrow, ncol), True
        return position, False  # Boundary check: cannot leave the grid.

    raise ValueError(f"Unknown action: {action}")


# --------------------------------------------------------------------------
# Agent function: rule -> action -> environment step
# --------------------------------------------------------------------------
def agent_program(percept: Dict[str, object], rules: List[Rule]) -> str:
    """Core reactive agent function: percept -> state -> rule -> action.

    Interprets the percept, matches it against the rule set, and returns the
    action prescribed by the first matching rule (or NO_OP if none match).
    The agent has no memory beyond the visited set carried in the percept; it
    reacts directly to the current percept only.
    """
    state = interpret_input(percept)
    rule = rule_match(rules, state)
    if rule is None:
        return "NO_OP"
    return rule_to_action(rule)


# --------------------------------------------------------------------------
# Environment: perception and rendering
# --------------------------------------------------------------------------
def get_percept(position: Position, grid: List[List[str]],
                visited: set) -> Dict[str, object]:
    """Build the percept observed by the agent at its current position."""
    row, col = position
    return {
        "position": position,
        "status": grid[row][col],
        "visited": set(visited),
    }


def print_grid(grid: List[List[str]]) -> None:
    """Print the current 2x2 environment grid state."""
    for row in grid:
        print("[" + "] [".join(row) + "]")


def all_clean(grid: List[List[str]]) -> bool:
    """Goal test: True when every cell in the grid is CLEAN."""
    return all(cell == CLEAN for row in grid for cell in row)


# --------------------------------------------------------------------------
# Agent-environment loop
# --------------------------------------------------------------------------
def run() -> None:
    """Run the reactive vacuum agent step-by-step until the goal or limit."""
    # Initial environment: all cells DIRTY, agent starts at (0, 0).
    grid: List[List[str]] = [[DIRTY for _ in range(GRID_SIZE)]
                             for _ in range(GRID_SIZE)]
    position: Position = (0, 0)
    visited: set = {position}
    rules = build_rule_set()
    steps = 0

    while steps < MAX_STEPS and not all_clean(grid):
        steps += 1

        # 1. Perception: the environment generates the current percept.
        percept = get_percept(position, grid, visited)

        # 2. State: the agent interprets the percept.
        state = interpret_input(percept)

        # 3. Rule matching: select the first applicable rule.
        rule = rule_match(rules, state)

        # 4. Agent function: derive the action from the matched rule.
        if rule is None:
            action = "NO_OP"
        else:
            action = rule_to_action(rule)

        rule_text = rule[0] if rule is not None else "No rule matched -> NO_OP"

        print(f"Step {steps}:")
        print(f"Position: {position}")
        print(f"Percept: {{'position': {percept['position']}, "
              f"'status': '{percept['status']}'}}")
        print(f"State: {{'position': {state['position']}, "
              f"'status': '{state['status']}'}}")
        print(f"Matched Rule: {rule_text}")
        print(f"Action: {action}")

        # 5. Action execution: the environment reacts to the action.
        if action != "NO_OP":
            position, moved = execute_action(action, position, grid)
            if moved:
                visited.add(position)

        print("Grid after action:")
        print_grid(grid)
        print()  # Blank line between steps.

    if all_clean(grid):
        print(f"Goal achieved: all cells are CLEAN. Total steps taken: {steps}")
    else:
        print(f"Goal NOT achieved within the step limit "
              f"({MAX_STEPS} steps). Total steps taken: {steps}")


if __name__ == "__main__":
    run()
