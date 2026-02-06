from time import sleep
from os import system, name
import sys
import random


def clear():
    if name == 'nt':
        system('cls')
    else:
        system('clear')


def dice_roll(sides):
    return random.randint(1, sides)


class Character:
    def __init__(self, char_class):
        self.char_class = char_class
        self.stats = self._initialize_stats()
        self.equipment = self._initialize_equipment()
        self.hp = 20 + self.stats['constitution']
        self.max_hp = self.hp
        
    def _initialize_stats(self):
        base_stats = {
            'strength': dice_roll(6),
            'dexterity': dice_roll(6),
            'constitution': dice_roll(6),
            'faith': 0 if self.char_class == 'Leper' else (5 if self.char_class == 'Witch Doctor' else dice_roll(6)),
            'charisma': 0 if self.char_class == 'Leper' else (dice_roll(6) if self.char_class == 'Witch Doctor' else 5)
        }
        return base_stats
    
    def _initialize_equipment(self):
        equipment_data = {
            'Leper': {
                'weapons': ['Shiv', 'Fist'],
                'armor': 'Leper Skin Leather',
                'weapon_stats': {'Shiv': 5, 'Fist': 1},
                'armor_speed': 10
            },
            'Witch Doctor': {
                'weapons': ['Blessed Molotov', 'Fist'],
                'armor': 'Witch Doctor Robes',
                'weapon_stats': {'Blessed Molotov': 2, 'Fist': 1},
                'armor_speed': 7
            },
            'Guardsmen': {
                'weapons': ['Flintlock Pistol'],
                'armor': 'Guardman Armor',
                'weapon_stats': {'Flintlock Pistol': 8},
                'armor_speed': 5
            }
        }
        return equipment_data[self.char_class]
    
    def get_speed(self):
        return self.equipment['armor_speed'] + self.stats['dexterity']
    
    def calculate_damage(self, weapon):
        weapon_lower = weapon.lower()
        base_damage = 0
        
        if 'shiv' in weapon_lower:
            base_damage = self.equipment['weapon_stats']['Shiv'] + self.stats['dexterity']
        elif 'fist' in weapon_lower:
            base_damage = self.equipment['weapon_stats']['Fist'] + self.stats['strength']
        elif 'molotov' in weapon_lower:
            base_damage = self.equipment['weapon_stats']['Blessed Molotov'] * self.stats['faith'] + self.stats['dexterity']
        elif 'flintlock' in weapon_lower or 'pistol' in weapon_lower:
            base_damage = self.equipment['weapon_stats']['Flintlock Pistol'] + self.stats['strength'] + self.stats['dexterity']
        
        return base_damage
    
    def increase_stat(self, stat_name, amount=2):
        if stat_name in self.stats:
            self.stats[stat_name] += amount
            return True
        return False
    
    def heal(self, amount):
        self.hp = min(self.hp + amount, self.max_hp)
    
    def display_stats(self):
        print("\n\t\t-Stats-\n")
        print("Strength\tDexterity\tConstitution\tFaith\tCharisma")
        print(f"{self.stats['strength']}\t\t{self.stats['dexterity']}\t\t{self.stats['constitution']}\t\t{self.stats['faith']}\t{self.stats['charisma']}")
    
    def display_inventory(self):
        print("\n\t\t-Inventory-\n")
        print(f"Weapons: {', '.join(self.equipment['weapons'])}")
        print(f"Armor: {self.equipment['armor']}")


class Monster:
    def __init__(self, name, hp, damage):
        self.name = name
        self.hp = hp
        self.max_hp = hp
        self.damage = damage
    
    def attack(self, player, hit_threshold):
        hit_chance = dice_roll(12)
        if hit_chance >= hit_threshold:
            player.hp -= self.damage
            sleep(1)
            print(f"\nThe {self.name} has attacked you for {self.damage} DMG!")
            return True
        else:
            sleep(1)
            print(f"\n{self.name} Missed!")
            return False
    
    def take_damage(self, amount):
        self.hp -= amount
        return self.hp <= 0


class Level:
    def __init__(self, level_num, player_class):
        self.level_num = level_num
        self.monsters = self._get_monsters(player_class)
        
    def _get_monsters(self, player_class):
        monster_data = {
            'Leper': [
                ('Gutter Rat', 5, 1),
                ('Rabid Dog', 15, 3),
                ('Cultist', 25, 5),
                ('Guardsmen Soldier', 30, 5),
                ('Guardsmen Captain', 50, 10)
            ],
            'Witch Doctor': [
                ('Gutter Rat', 5, 1),
                ('Rabid Dog', 15, 3),
                ('Guardsmen Soldier', 30, 5),
                ('Leper', 30, 5),
                ('Dark Spirit Gerhard', 50, 10)
            ],
            'Guardsmen': [
                ('Gutter Rat', 5, 1),
                ('Rabid Dog', 15, 3),
                ('Cultist', 25, 5),
                ('Leper', 30, 5),
                ('Leper Captain', 50, 10)
            ]
        }
        
        monsters = []
        for name, hp, damage in monster_data[player_class]:
            monsters.append(Monster(name, hp, damage))
        return monsters
    
    def get_current_monster(self):
        return self.monsters[self.level_num - 1]


class Game:
    def __init__(self):
        self.player = None
        self.current_level = 1
        
    def character_creation(self):
        clear()
        print("\tWelcome to Leprosy\n")
        
        while True:
            print("~~~~Class Menu~~~~")
            print("1. Leper")
            print("2. Witch Doctor")
            print("3. Guardsmen")
            
            choice = input("Enter your class choice [1-3]: ")
            
            class_map = {'1': 'Leper', '2': 'Witch Doctor', '3': 'Guardsmen'}
            
            if choice in class_map:
                self.player = Character(class_map[choice])
                print(f"\nYou have chosen to play as a {self.player.char_class}")
                sleep(2)
                
                self.player.display_stats()
                sleep(2)
                
                print(f"\nStrength modifies Shiv, Fist, and Flintlock Pistol DMG")
                print(f"Dexterity modifies Shiv, Molotov, and Flintlock Pistol DMG")
                print(f"Constitution determines your starting HP & HP gained per level")
                print(f"Faith modifies blessed weapons")
                print(f"Charisma modifies enemy hit chance")
                sleep(3)
                
                self.player.display_inventory()
                print(f"\nHealth Points: {self.player.hp}")
                sleep(2)
                
                confirm = input(f"\nContinue with these stats as a {self.player.char_class}? [y/n]: ").lower()
                if confirm == 'y':
                    break
                else:
                    clear()
            else:
                print("***INVALID ENTRY***")
    
    def show_intro_dialog(self):
        clear()
        intros = {
            'Leper': [
                "*Above you lies a village, a village you belonged to before your exile to the tunnels*",
                "*Surrounding you are shadows of hunched over figures, decaying, dying, and timid*",
                "Once apart of the civilization above, they now lay in waste",
                "No one down here has any identification documents, as they were destroyed with their belongings once they were exposed as a leper",
                "You have the same past, even a name. But its your word and no one cares to listen",
                "Anonymity provides its advantages, you have been tasked by your leader to commit a great act of revolution",
                "Slay the rulers of the village above to restore your livelyhood, for what little time you have left",
                "*You leave the tunnels*"
            ],
            'Witch Doctor': [
                "*You're surrounded by rock walls with scripture written in ash*",
                "*A cultist crouched next to you begins to draw mysterious symbols by your feet*",
                "*The cultist looks up at you in awe*",
                "'Doctor, you have been born again in the realm between the conscious and the material'",
                "'What you experience here are senses outside of your fleshy vessel'",
                "'Carry on and save humanity from their flesh and god will praise your eternal name'",
                "*You leave the cave*"
            ],
            'Guardsmen': [
                "*You walk into the Captains Quarters, dawning your new guardsmen uniform*",
                "Your captain has tasked you to investigate the disappeareance of the village doctor",
                "They've been gone for sometime but the captain refuses to quantify the amount of time",
                "Details are slim and confidential",
                "*You descend down the village steps*"
            ]
        }
        
        for line in intros[self.player.char_class]:
            print(line)
            sleep(2)
    
    def combat(self, monster):
        while self.player.hp > 0 and monster.hp > 0:
            clear()
            print(f"\nBattle: {monster.name.upper()}")
            print(f"Monster HP: {monster.hp}/{monster.max_hp} | Damage: {monster.damage}")
            print(f"Your HP: {self.player.hp}/{self.player.max_hp}\n")
            sleep(1)
            
            monster.attack(self.player, self.player.stats['charisma'])
            
            if self.player.hp <= 0:
                print("\nYou have Died")
                sleep(3)
                return False
            
            print(f"\nYour HP: {self.player.hp}")
            sleep(1)
            
            print(f"\nAvailable weapons: {', '.join(self.player.equipment['weapons'])}")
            weapon_choice = input("\nWhich weapon are you attacking with? ").lower()
            
            damage = self.player.calculate_damage(weapon_choice)
            
            if damage > 0:
                print(f"\nYou wield your {weapon_choice}")
                sleep(1)
                
                if self.player.char_class == 'Leper':
                    monster.take_damage(damage)
                    print(f"You hit the {monster.name} for {damage} DMG!")
                    sleep(1)
                    print(f"The {monster.name} has {monster.hp} HP!")
                    monster.take_damage(damage)
                    print(f"You hit the {monster.name} for {damage} DMG!")
                    sleep(1)
                    print(f"The {monster.name} has {monster.hp} HP!")
                elif self.player.char_class == 'Guardsmen' and 'flintlock' in weapon_choice:
                    monster.take_damage(damage)
                    print(f"You hit the {monster.name} for {damage} DMG!")
                    sleep(1)
                    print(f"The {monster.name} has {monster.hp} HP!")
                    if monster.hp > 0:
                        print("\nYou need to reload")
                        sleep(1)
                        print(f"The {monster.name} attacks while you reload!")
                        monster.attack(self.player, self.player.stats['charisma'])
                else:
                    monster.take_damage(damage)
                    print(f"You hit the {monster.name} for {damage} DMG!")
                    sleep(1)
                    print(f"The {monster.name} has {monster.hp} HP!")
            else:
                print("Invalid weapon choice. You do nothing this turn.")
                sleep(2)
            
            sleep(1)
        
        if monster.hp <= 0:
            print(f"\nThe {monster.name} is dead!")
            return True
        
        return False
    
    def level_reward(self):
        print("\n[1] STAT INCREASE by 2")
        print("[2] HEAL 10 HP")
        
        while True:
            try:
                choice = int(input("\nChoose an option [1/2]: "))
                if choice == 1:
                    self.player.display_stats()
                    print("\n1.Strength  2.Dexterity  3.Constitution  4.Faith  5.Charisma")
                    
                    stat_choice = int(input("Select a STAT to increase [1/2/3/4/5]: "))
                    stat_map = {1: 'strength', 2: 'dexterity', 3: 'constitution', 4: 'faith', 5: 'charisma'}
                    
                    if stat_choice in stat_map:
                        stat_name = stat_map[stat_choice]
                        self.player.increase_stat(stat_name, 2)
                        print(f"Your {stat_name.capitalize()} has increased by 2")
                        self.player.display_stats()
                        break
                    else:
                        print("Invalid choice")
                elif choice == 2:
                    self.player.heal(10)
                    print(f"Your HP is now: {self.player.hp} HP")
                    break
                else:
                    print("Invalid choice")
            except ValueError:
                print("Please enter a number")
        
        self.player.max_hp += self.player.stats['constitution']
        self.player.hp += self.player.stats['constitution']
        sleep(1)
        print(f"Your HP has been increased by your Constitution!")
    
    def play_level(self, level_num):
        level = Level(level_num, self.player.char_class)
        monster = level.get_current_monster()
        
        clear()
        
        if level_num < 5:
            monster_speeds = {1: 9, 2: 11, 3: 17, 4: 17}
            speed_threshold = monster_speeds.get(level_num, 15)
            
            action = input(f"\nThe {monster.name} appears before you.\nRun or Attack? ").lower()
            
            if action == 'run':
                if self.player.get_speed() > speed_threshold:
                    print(f"\nYou dash past the {monster.name}, bigger threats await you")
                    sleep(2)
                    return True
                else:
                    print(f"\nNot fast enough, The {monster.name} is right behind you")
                    sleep(2)
        else:
            input(f"\nThe {monster.name} draws their weapon. Press Enter to Attack...")
        
        if self.combat(monster):
            self.level_reward()
            return True
        else:
            return False
    
    def run(self):
        self.character_creation()
        
        confirm = input(f"\nWould you like to start your journey as a {self.player.char_class}? [y/n]: ").lower()
        if confirm != 'y':
            print("Restarting...")
            return
        
        self.show_intro_dialog()
        
        for level_num in range(1, 6):
            sleep(2)
            if not self.play_level(level_num):
                retry = input(f"\nRestart level {level_num}? [y/n]: ").lower()
                if retry == 'y':
                    self.player.hp = self.player.max_hp
                    if not self.play_level(level_num):
                        print("Game Over")
                        return
                else:
                    print("Game Over")
                    return
            
            if level_num < 5:
                cont = input(f"\nContinue to next level? [y/n]: ").lower()
                if cont != 'y':
                    print("Game paused")
                    return
        
        clear()
        print(f"\n\nWell done {self.player.char_class}, your journey ends here. For now.")
        sleep(5)
        sys.exit("\nGoodbye!")


if __name__ == "__main__":
    game = Game()
    game.run()
