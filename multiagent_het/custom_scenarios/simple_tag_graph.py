from typing import Optional, Tuple, List
import numpy as np
from scipy import sparse
from numpy import ndarray as arr
from onpolicy_het.envs.mpe.core import World, Agent, Landmark
from onpolicy_het.envs.mpe.scenario import BaseScenario

from multiagent_het.core import World, Agent, Landmark, Entity
from multiagent_het.scenario import BaseScenario

entity_mapping = {"agent": 0, "landmark": 1, "obstacle": 2}


class Scenario(BaseScenario):
    def make_world(self, args):

        world = World()
        
        # NEW
        # graph related attributes
        world.cache_dists = True  # cache distance between all entities
        world.graph_mode = True
        world.graph_feat_type = args.graph_feat_type
        world.world_length = args.episode_length
        
        # set any world properties first
        world.dim_c = 2
        num_good_agents = args.num_good_agents  # 1
        num_adversaries = args.num_adversaries  # 3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = args.num_landmarks  # 2
        self.num_obstacles = args.num_obstacles
        world.current_time_step = 0

        # pull params from args or computed
        self.world_size = args.world_size
        self.num_agents = num_agents
        self.num_scripted_agents = args.num_scripted_agents
        self.max_speed = args.max_speed
        if not hasattr(args, "max_edge_dist"):
            self.max_edge_dist = 1
            print("_" * 60)
            print(
                f"Max Edge Distance for graphs not specified. "
                f"Setting it to {self.max_edge_dist}"
            )
            print("_" * 60)
        else:
            self.max_edge_dist = args.max_edge_dist

        # add agents
        global_id = 0
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            print(f'make_world::agent:{i}')
            agent.id = i
            agent.name = f"agent {i}"
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.global_id = global_id
            global_id += 1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            print(f'make_world::agent:{i}')
            landmark.id = i
            landmark.name = f"landmark {i}"
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
            landmark.global_id = global_id
            global_id += 1
        # add obstacles
        world.obstacles = [Landmark() for i in range(self.num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.id = i
            obstacle.name = f"obstacle {i}"
            obstacle.collide = True
            obstacle.movable = False
            obstacle.global_id = global_id
            global_id += 1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world_old(self, world):
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.assign_landmark_colors()
        # random properties for landmarks
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def reset_world(self, world: World) -> None:
        # metrics to keep track of
        world.current_time_step = 0
        # to track time required to reach goal
        world.times_required = -1 * np.ones(self.num_agents)
        # track distance left to the goal
        world.dist_left_to_goal = -1 * np.ones(self.num_agents)
        # number of times agents collide with stuff
        world.num_obstacle_collisions = np.zeros(self.num_agents)
        world.num_agent_collisions = np.zeros(self.num_agents)

        #################### set colours ####################
        # set colours for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # set colours for scripted agents
        for i, agent in enumerate(world.scripted_agents):
            agent.color = np.array([0.15, 0.15, 0.15])
        # set colours for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.85, 0.15])
        # set colours for scripted agents goals
        for i, landmark in enumerate(world.scripted_agents_goals):
            landmark.color = np.array([0.15, 0.95, 0.15])
        # set colours for obstacles
        for i, obstacle in enumerate(world.obstacles):
            obstacle.color = np.array([0.25, 0.25, 0.25])
        #####################################################
        self.random_scenario(world)


    def random_scenario(self, world):
        """
        Randomly place agents and landmarks
        """
        ####### set random positions for entities ###########
        # set random static obstacles first
        for obstacle in world.obstacles:
            obstacle.state.p_pos = 0.8 * np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            obstacle.state.p_vel = np.zeros(world.dim_p)
        #####################################################

        # set agents at random positions not colliding with obstacles
        num_agents_added = 0
        agents_added = []
        while True:
            if num_agents_added == self.num_agents:
                break
            random_pos = np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            agent_size = world.agents[num_agents_added].size
            obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
            agent_collision = self.check_agent_collision(
                random_pos, agent_size, agents_added
            )
            if not obs_collision and not agent_collision:
                world.agents[num_agents_added].state.p_pos = random_pos
                world.agents[num_agents_added].state.p_vel = np.zeros(world.dim_p)
                world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
                agents_added.append(world.agents[num_agents_added])
                num_agents_added += 1
        #####################################################

        # set scripted agents goals at random positions not colliding with obstacles
        num_scripted_agents_added = 0
        while True:
            if num_scripted_agents_added == self.num_scripted_agents:
                break
            random_pos = np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            agent_size = world.scripted_agents[num_scripted_agents_added].size
            obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
            agent_collision = self.check_agent_collision(
                random_pos, agent_size, agents_added
            )
            if not obs_collision and not agent_collision:
                world.scripted_agents[
                    num_scripted_agents_added
                ].state.p_pos = random_pos
                world.scripted_agents[num_scripted_agents_added].state.p_vel = np.zeros(
                    world.dim_p
                )
                world.scripted_agents[num_scripted_agents_added].state.c = np.zeros(
                    world.dim_c
                )
                agents_added.append(world.scripted_agents[num_scripted_agents_added])
                num_scripted_agents_added += 1
        #####################################################

        ############ find minimum times to goals ############
        #if self.max_speed is not None:
        #    for agent in world.agents:
        #        self.min_time(agent, world)
        #####################################################
        ############ update the cached distances ############
        world.calculate_distances()
        self.update_graph(world)
        ####################################################

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    # check collision of entity with obstacles
    def is_obstacle_collision(self, pos, entity_size: float, world: World) -> bool:
        # pos is entity position "entity.state.p_pos"
        collision = False
        for obstacle in world.obstacles:
            delta_pos = obstacle.state.p_pos - pos
            dist = np.linalg.norm(delta_pos)
            dist_min = obstacle.size + entity_size
            if dist < dist_min:
                collision = True
                break
        return collision

    # check collision of agent with other agents
    def check_agent_collision(self, pos, agent_size, agent_added) -> bool:
        collision = False
        if len(agent_added):
            for agent in agent_added:
                delta_pos = agent.state.p_pos - pos
                dist = np.linalg.norm(delta_pos)
                if dist < (agent.size + agent_size):
                    collision = True
                    break
        return collision

    # get min time required to reach to goal without obstacles
    def min_time(self, agent: Agent, world: World) -> float:
        assert agent.max_speed is not None, "Agent needs to have a max_speed"
        agent_id = agent.id
        # get the goal associated to this agent
        landmark = world.get_entity(entity_type="landmark", id=agent_id)
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
        min_time = dist / agent.max_speed
        agent.goal_min_time = min_time
        return min_time

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False  # different from openai
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False  # different from openai
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min(
                    [
                        np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                        for a in agents
                    ]
                )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )
    
    #########################################################################

    def done(self, agent: Agent, world: World) -> bool:
        # done is False if done_callback is not passed to
        # environment.MultiAgentEnv
        # This is same as original version
        # Check `_get_done()` in environment.MultiAgentEnv
        return False

    def graph_observation(self, agent: Agent, world: World) -> Tuple[arr, arr]:
        """
        FIXME: Take care of the case where edge_list is empty
        Returns: [node features, adjacency matrix]
        • Node features (num_entities, num_node_feats):
            If `global`:
                • node features are global [pos, vel, goal, entity-type]
                • edge features are relative distances (just magnitude)
                NOTE: for `landmarks` and `obstacles` the `goal` is
                        the same as its position
            If `relative`:
                • node features are relative [pos, vel, goal, entity-type] to ego agents
                • edge features are relative distances (just magnitude)
                NOTE: for `landmarks` and `obstacles` the `goal` is
                        the same as its position
        • Adjacency Matrix (num_entities, num_entities)
            NOTE: using the distance matrix, need to do some post-processing
            If `global`:
                • All close-by entities are connectd together
            If `relative`:
                • Only entities close to the ego-agent are connected

        """
        num_entities = len(world.entities)
        print(f'num_entities:{num_entities}')
        for i, entity in enumerate(world.entities):
            print(f'entity[{i}]:{entity.id} - {entity.name}')

        # node observations
        node_obs = []
        if world.graph_feat_type == "global":
            for i, entity in enumerate(world.entities):
                print(f'entity:{entity}')
                node_obs_i = self._get_entity_feat_global(entity, world)
                node_obs.append(node_obs_i)
        elif world.graph_feat_type == "relative":
            for i, entity in enumerate(world.entities):
                node_obs_i = self._get_entity_feat_relative(agent, entity, world)
                node_obs.append(node_obs_i)

        print(f'node_obs:{node_obs}')
        node_obs = np.array(node_obs)
        adj = world.cached_dist_mag

        return node_obs, adj

    def info_callback(self, agent: Agent, world: World) -> Tuple:
        # TODO modify this
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        #goal = world.get_entity("landmark", agent.id)
        #dist = np.sqrt(np.sum(np.square(agent.state.p_pos - goal.state.p_pos)))
        #world.dist_left_to_goal[agent.id] = dist
        # only update times_required for the first time it reaches the goal
        #if dist < self.min_dist_thresh and (world.times_required[agent.id] == -1):
        #    world.times_required[agent.id] = world.current_time_step * world.dt

        if agent.collide:
            if self.is_obstacle_collision(agent.state.p_pos, agent.size, world):
                world.num_obstacle_collisions[agent.id] += 1
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(agent, a):
                    world.num_agent_collisions[agent.id] += 1

        agent_info = {
            "Dist_to_goal": world.dist_left_to_goal[agent.id],
            "Time_req_to_goal": world.times_required[agent.id],
            # NOTE: total agent collisions is half since we are double counting
            "Num_agent_collisions": world.num_agent_collisions[agent.id],
            "Num_obst_collisions": world.num_obstacle_collisions[agent.id],
        }
        if self.max_speed is not None:
            agent_info["Min_time_to_goal"] = agent.goal_min_time
        return agent_info

    def get_id(self, agent: Agent) -> arr:
        return np.array([agent.global_id])

    def update_graph(self, world: World):
        """
        Construct a graph from the cached distances.
        Nodes are entities in the environment
        Edges are constructed by thresholding distances
        """
        dists = world.cached_dist_mag
        # just connect the ones which are within connection
        # distance and do not connect to itself
        connect = np.array((dists <= self.max_edge_dist) * (dists > 0)).astype(int)
        sparse_connect = sparse.csr_matrix(connect)
        sparse_connect = sparse_connect.tocoo()
        row, col = sparse_connect.row, sparse_connect.col
        edge_list = np.stack([row, col])
        world.edge_list = edge_list
        if world.graph_feat_type == "global":
            world.edge_weight = dists[row, col]
        elif world.graph_feat_type == "relative":
            world.edge_weight = dists[row, col]

    def _get_entity_feat_global(self, entity: Entity, world: World) -> arr:
        """
        Returns: ([velocity, position, goal_pos, entity_type])
        in global coords for the given entity
        Note: This environment should not have any landmark
        """
        pos = entity.state.p_pos
        vel = entity.state.p_vel
        print(f'entity:{entity.name} {entity.id}')

        if "agent" in entity.name:
            #goal_pos = world.get_entity("landmark", entity.id).state.p_pos
            goal_pos = pos
            entity_type = entity_mapping["agent"]
        elif "landmark" in entity.name:
            goal_pos = pos
            entity_type = entity_mapping["landmark"]
        elif "obstacle" in entity.name:
            goal_pos = pos
            entity_type = entity_mapping["obstacle"]
        else:
            raise ValueError(f"{entity.name} not supported")

        print(f'vel, pos, goal_pos, entity_type:{vel} {pos} {goal_pos} {entity_type}')
        return np.hstack([vel, pos, goal_pos, entity_type])

    def _get_entity_feat_relative(
        self, agent: Agent, entity: Entity, world: World
    ) -> arr:
        """
        Returns: ([velocity, position, goal_pos, entity_type])
        in coords relative to the `agent` for the given entity
        """
        agent_pos = agent.state.p_pos
        agent_vel = agent.state.p_vel
        entity_pos = entity.state.p_pos
        entity_vel = entity.state.p_vel
        rel_pos = entity_pos - agent_pos
        rel_vel = entity_vel - agent_vel
        if "agent" in entity.name:
            goal_pos = world.get_entity("landmark", entity.id).state.p_pos
            rel_goal_pos = goal_pos - agent_pos
            entity_type = entity_mapping["agent"]
        elif "landmark" in entity.name:
            rel_goal_pos = rel_pos
            entity_type = entity_mapping["landmark"]
        elif "obstacle" in entity.name:
            rel_goal_pos = rel_pos
            entity_type = entity_mapping["obstacle"]
        else:
            raise ValueError(f"{entity.name} not supported")

        return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type])


# actions: [None, ←, →, ↓, ↑, comm1, comm2]
if __name__ == "__main__":
    from multiagent_het.environment import MultiAgentGraphEnv
    from multiagent_het.policy import InteractivePolicy

    # makeshift argparser
    class Args:
        def __init__(self):
            self.num_agents: int = 4
            self.collaborative: bool = False
            self.max_speed: Optional[float] = 2
            self.collision_rew: float = 5
            self.goal_rew: float = 5
            self.min_dist_thresh: float = 0.1
            self.use_dones: bool = False
            self.episode_length: int = 25
            self.num_good_agents: int = 1
            self.num_adversaries: int = 3
            self.num_landmarks: int = 0
            self.num_obstacles: int = 2
            self.max_edge_dist: float = 1
            self.graph_feat_type: str = "global"
            self.world_size: int = 2
            self.num_scripted_agents: int = 0

    args = Args()

    scenario = Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentGraphEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        graph_observation_callback=scenario.graph_observation,
        info_callback=scenario.info_callback,
        done_callback=scenario.done,
        id_callback=scenario.get_id,
        update_graph=scenario.update_graph,
        shared_viewer=False,
    )
    # render call to create viewer window
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n, agent_id_n, node_obs_n, adj_n = env.reset()
    stp = 0
    while True:
        # query for action from each agent's policy
        act_n = []
        dist_mag = env.world.cached_dist_mag

        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        # print(act_n)
        obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(act_n)
        # print(obs_n[0].shape, node_obs_n[0].shape, adj_n[0].shape)

        # render all agent views
        env.render()
        stp += 1
        # display rewards
