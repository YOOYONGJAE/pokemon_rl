import pygame

def _init_pygame_core(env):
    pygame.init()
    pygame.display.set_caption("Pokemon Battle")
    env.screen = pygame.display.set_mode(env.window_size)
    env.clock = pygame.time.Clock()

    # 포켓몬 이미지들
    env.my_img = pygame.image.load("assets/Bulbasaur.png").convert_alpha()
    env.opp_img = pygame.image.load("assets/Snorlax.png").convert_alpha() 

    env.my_img  = pygame.transform.scale(env.my_img,  (100, 80))
    env.opp_img = pygame.transform.scale(env.opp_img, (100, 100))        

    env.font = pygame.font.Font(None, 25)

def render_core(env):

    if env.render_mode is None:
        print(f"Turn {env.turn_count}")
        print(f"  My HP:  {env.my_hp:3d}, Def: {env.my_def:3.0f}, "
            f"PP: {env.my_pp}, AccBuff: {env.my_acc_buff_stack}")
        print(f"  Opp HP: {env.opp_hp:3d}, Def: {env.opp_def:3.0f}, "
            f"PP: {env.opp_pp}, AccDebuff: {env.opp_acc_debuff_stack}")
        print(f"My move counts:", getattr(env, "my_move_count", None))
        print("-" * 50)

    else:
        # "휴먼"체크
        if env.render_mode != "human":
            return

        if env.screen is None:
            env._init_pygame()

        # pygame 이벤트 처리 (창 닫기 등)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        env.screen.fill((255, 255, 255))

        # 내 포켓몬
        my_x, my_y = 100, 300
        env.screen.blit(env.my_img, (my_x, my_y))
        # 상대 포켓몬
        opp_x, opp_y = 420, 100
        env.screen.blit(env.opp_img, (opp_x, opp_y))

        bar_width = 200
        bar_height = 15

        my_hp_ratio = env.my_hp / env.MAX_HP
        opp_hp_ratio = env.opp_hp / env.MAX_HP

        my_hp_text = f"{env.my_hp} / {env.MAX_HP}"
        opp_hp_text = f"{env.opp_hp} / {env.MAX_HP}"

        my_hp_surf = env.font.render(my_hp_text, True, (0, 0, 0))
        opp_hp_surf = env.font.render(opp_hp_text, True, (0, 0, 0))

        env.screen.blit(my_hp_surf, (my_x -70, my_y -20))
        env.screen.blit(opp_hp_surf, (opp_x -80, opp_y -20))

        # 배틀 시작, 종료 메시지
        if getattr(env, "battle_ended", False):
            status_text = "BATTLE END"
        else:
            status_text = "BATTLE START"

        status_surf = env.font.render(status_text, True, (0, 0, 0))
        env.screen.blit(status_surf, (10, 10))  

        cur_ep   = getattr(env, "current_episode", 0)
        total_ep = getattr(env, "total_episodes", 0)
        cur_step = getattr(env, "current_step", 0)        

        ep_text   = f"Episode: {cur_ep}/{total_ep}" if total_ep > 0 else f"Episode: {cur_ep}"
        step_text = f"Step: {cur_step}"            
        ep_surf   = env.font.render(ep_text,   True, (0, 0, 0))
        step_surf = env.font.render(step_text, True, (0, 0, 0))

        env.screen.blit(ep_surf,   (10, 30))
        env.screen.blit(step_surf, (10, 50))

        wins   = getattr(env, "wins", 0)
        losses = getattr(env, "losses", 0)
        draws  = getattr(env, "draws", 0)
        win_rt = getattr(env, "win_rate", 0.0)
        avg_r  = getattr(env, "avg_reward", 0.0)

        wld_text   = f"W/L/D: {wins} / {losses} / {draws}"
        stats_text = f"WinRate: {win_rt:.3f}  AvgR: {avg_r:.3f}"

        wld_surf   = env.font.render(wld_text,   True, (0, 0, 0))
        stats_surf = env.font.render(stats_text, True, (0, 0, 0))

        env.screen.blit(wld_surf,   (10, 70))
        env.screen.blit(stats_surf, (10, 90))

        move_counts = getattr(env, "my_move_count", [0, 0, 0, 0])
        y0 = 120
        for i, name in enumerate(env.my_move_names):
            text  = f"{name}: {move_counts[i]} USED"
            surf  = env.font.render(text, True, (0, 0, 0))
            env.screen.blit(surf, (10, y0 + i * 18))

        pygame.draw.rect(env.screen, (0, 255, 0), (my_x, my_y - 20, int(bar_width * my_hp_ratio), bar_height))
        pygame.draw.rect(env.screen, (0, 255, 0), (opp_x, opp_y - 20, int(bar_width * opp_hp_ratio), bar_height))

        my_move_name  = getattr(env, "last_my_move_name", "???")
        opp_move_name = getattr(env, "last_opp_move_name", "???")
        
        my_move_text  = f"My Bulbasaur used [{my_move_name}]!!"
        opp_move_text = f"The opposing Snorlax used [{opp_move_name}]!!"

        my_move_surf  = env.font.render(my_move_text,  True, (0, 0, 0))
        opp_move_surf = env.font.render(opp_move_text, True, (0, 0, 0))

        env.screen.blit(my_move_surf,  (my_x -60, my_y+100))
        env.screen.blit(opp_move_surf, (opp_x -120, opp_y+100))        

        pygame.display.flip()

        env.clock.tick(env.metadata["render_fps"])
    

def close_core(env):
    if env.screen is not None:
        pygame.display.quit()
        pygame.quit()
        env.screen = None
        env.clock = None