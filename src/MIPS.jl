# ===========================================================
#                        LOAD PACKAGES
# ===========================================================

# To install DataFrames, simply run Pkg.add("DataFrames")
using DataFrames

#Load MIP Package
using Gurobi
#using GLPKMathProgInterface, JLD

# Once again, to install run Pkg.add("JuMP")
using JuMP

# ================================================
#                    PARAMETERS
# ================================================
function main(date_list::Vector{Int},
              points_system_list::Vector{String},
              num_lineups_list::Vector{Int},
              num_overlap_list::Vector{Int},
              max_appearances_list::Vector{Int},
              alpha_list::Vector{Float64},
              path_to_proj::String)
              
    min_games = 2
    salary_cap = 50000
    write_lineups_file = true
    
    # Iterate over the date_list
    for date_string in date_list
        
        # Build the path to players data based on date_string
        path_to_players = string(path_to_proj, "/Model/Output/Model Input/Model Input", date_string, ".csv")

        # Load the player information
        player_info = readtable(path_to_players)

        # Run the algorithm for each combination of parameters
        for value1 in num_lineups_list
            for value3 in num_overlap_list
                for value4 in max_appearances_list
                    for value5 in alpha_list
                        for value6 in points_system_list
                            EM_algorithm(value6, value1, value3, value4, value5, value5 * 10, player_info, min_games, salary_cap, path_to_proj, date_string)
                        end
                    end
                end
            end
        end
    end
end



# ===========================================================
#                    LINEUP FUNCTION
# ===========================================================
function get_lineup(player_info, points_system, min_games, salary_cap,num_lineups,num_overlap, max_appearances, alpha)

    num_players = size(player_info,1);

    games = unique(player_info[:GameInfo])
    display(games)
    num_games = size(games)[1]

    game_info = zeros(Int, size(games)[1])

    # Populate player_info with the corresponding information for their teams
    for j=1:size(games)[1]
        if player_info[1, :GameInfo] == games[j]
            game_info[j] =1
        end
    end
    players_games = game_info'


    for i=2:num_players
        game_info = zeros(Int, size(games)[1])
        for j=1:size(games)[1]
            if player_info[i, :GameInfo] == games[j]
                game_info[j] =1
            end
        end
        players_games = vcat(players_games, game_info')
    end

    teams = unique(player_info[:TeamAbbrev])
    display(teams)
    num_teams = size(teams)[1]

    team_info = zeros(Int, size(teams)[1])

    # Populate player_info with the corresponding information for their teams
    for j=1:size(teams)[1]
        if player_info[1, :TeamAbbrev] == teams[j]
            team_info[j] =1
        end
    end
    players_teams = team_info'


    for i=2:num_players
        team_info = zeros(Int, size(teams)[1])
        for j=1:size(teams)[1]
            if player_info[i, :TeamAbbrev] == teams[j]
                team_info[j] =1
            end
        end
        players_teams = vcat(players_teams, team_info')
    end
    
    #Get points system, and limit to starters if applicable
    player_info[:Points] = player_info[Symbol(points_system)]+alpha*player_info[:StdDev]

    # create dataframes to store solutions
    lineup = DataFrame(Int,num_lineups,9)
    col_names_lineups = ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"]
    names!(lineup.colindex, map(parse, col_names_lineups))

    results = DataFrame(String,num_lineups,15)
    col_names_results = ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST", "points_system", "num_lineups", "num_overlap", "max_appearances", "alpha", "iteration"]
    names!(results.colindex, map(parse, col_names_results))

    # create dataframe for overlap constraints
    overlap = zeros(num_players,num_lineups)

    # create dataframe for objective values
    scores = DataFrame(Float64,num_lineups, 2)
    col_names = ["Objective", "Total Salary"]
    names!(scores, Symbol.(col_names))

    m = Model(solver=GurobiSolver(OutputFlag=0))
    #m = Model(solver=GLPKSolverMIP())

    # variable for players in lineup
    @variable(m, player_lineup[i=1:num_players,j=1:9], Bin)

    # variable to see if game is used
    @variable(m, game_lineup[k=1:num_games], Bin)

    # nine players constraint
    @constraint(m, sum(sum(player_lineup[i,j] for j=1:9) for i=1:num_players) == 9)

    # each player used at most once
    for i in 1:num_players
        @constraint(m, sum(player_lineup[i,j] for j=1:9) <= 1)
    end

    # salary constraint
    @constraint(m, sum(player_info[i,:Salary]*sum(player_lineup[i,j] for j=1:9) for i=1:num_players) <= salary_cap)

    # one QB
    @constraint(m, sum(player_lineup[i,1] for i=1:num_players) == 1)

    # one RB1
    @constraint(m, sum(player_lineup[i,2] for i=1:num_players) == 1 )

    # one RB2
    @constraint(m, sum(player_lineup[i,3] for i=1:num_players) == 1)

    # one WR1
    @constraint(m, sum(player_lineup[i,4] for i=1:num_players) == 1)

    # one WR2
    @constraint(m, sum(player_lineup[i,5] for i=1:num_players) == 1)

    # one WR3
    @constraint(m, sum(player_lineup[i,6] for i=1:num_players) == 1)

    # one TE
    @constraint(m, sum(player_lineup[i,7] for i=1:num_players) == 1)

    # one FLEX
    @constraint(m, sum(player_lineup[i,8] for i=1:num_players) == 1)

    # one DST
    @constraint(m, sum(player_lineup[i,9] for i=1:num_players) == 1)

    # each player only used for their position
    for i in 1:num_players
        @constraint(m, player_lineup[i,1] <= player_info[i,:QB])
        @constraint(m, player_lineup[i,2] <= player_info[i,:RB])
        @constraint(m, player_lineup[i,3] <= player_info[i,:RB])
        @constraint(m, player_lineup[i,4] <= player_info[i,:WR])
        @constraint(m, player_lineup[i,5] <= player_info[i,:WR])
        @constraint(m, player_lineup[i,6] <= player_info[i,:WR])
        @constraint(m, player_lineup[i,7] <= player_info[i,:TE])
        @constraint(m, player_lineup[i,8] <= player_info[i,:FLEX])
        @constraint(m, player_lineup[i,9] <= player_info[i,:DST])
    end

    # at least 2 different games
    #@variable(m, used_game[i=1:num_games], Bin)
    #@constraint(m, constr[i=1:num_games], used_game[i] <= sum(sum(players_games[t, i]*player_lineup[t,j] for t=1:num_players) for j=1:9))
    #@constraint(m, sum(used_game[i] for i=1:num_games) >= 2)

    #Constraint 1 - QB, WR1, TE1 from same team, WR2 not on same team
    @variable(m, used_team_qb[i=1:num_teams], Bin)
    @constraint(m, constr[i=1:num_teams, t=1:num_players], used_team_qb[i] >= players_teams[t, i]*player_lineup[t,1])
    @constraint(m, constr[i=1:num_teams, t=1:num_players], used_team_qb[i] >= players_teams[t, i]*player_lineup[t,4])
    @constraint(m, constr[i=1:num_teams, t=1:num_players], 1-used_team_qb[i] >= players_teams[t, i]*player_lineup[t,5])
    @constraint(m, constr[i=1:num_teams, t=1:num_players], used_team_qb[i] >= players_teams[t, i]*player_lineup[t,7])
    @constraint(m, sum(used_team_qb[i] for i=1:num_teams) <= 1)

    #Constraint 1 - QB, WR1, TE1, WR2 from same game. No other players in this game
    @variable(m, used_game_qb[i=1:num_games], Bin)
    @constraint(m, constr[i=1:num_games, t=1:num_players], used_game_qb[i] >= players_games[t, i]*player_lineup[t,1])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_qb[i] >= players_games[t, i]*player_lineup[t,2])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_qb[i] >= players_games[t, i]*player_lineup[t,3])
    @constraint(m, constr[i=1:num_games, t=1:num_players], used_game_qb[i] >= players_games[t, i]*player_lineup[t,4])
    @constraint(m, constr[i=1:num_games, t=1:num_players], used_game_qb[i] >= players_games[t, i]*player_lineup[t,5])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_qb[i] >= players_games[t, i]*player_lineup[t,6])
    @constraint(m, constr[i=1:num_games, t=1:num_players], used_game_qb[i] >= players_games[t, i]*player_lineup[t,7])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_qb[i] >= players_games[t, i]*player_lineup[t,8])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_qb[i] >= players_games[t, i]*player_lineup[t,9])
    @constraint(m, sum(used_game_qb[i] for i=1:num_games) <= 1)

    #Constraint 2 - DST and RB1 can be in same game. No other players in same game as DST
    @variable(m, used_game_dst[i=1:num_games], Bin)
    @variable(m, used_game_rb1[i=1:num_games], Bin)
    @constraint(m, constr[i=1:num_games, t=1:num_players], used_game_dst[i] >= players_games[t, i]*player_lineup[t,9])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_dst[i] >= players_games[t, i]*player_lineup[t,1])
    @constraint(m, constr[i=1:num_games, t=1:num_players], used_game_rb1[i] >= players_games[t, i]*player_lineup[t,2])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_dst[i] >= players_games[t, i]*player_lineup[t,3])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_dst[i] >= players_games[t, i]*player_lineup[t,4])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_dst[i] >= players_games[t, i]*player_lineup[t,5])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_dst[i] >= players_games[t, i]*player_lineup[t,6])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_dst[i] >= players_games[t, i]*player_lineup[t,7])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_dst[i] >= players_games[t, i]*player_lineup[t,8])
    @constraint(m, sum(used_game_dst[i] for i=1:num_games) <= 1)
    @constraint(m, sum(used_game_rb1[i] for i=1:num_games) <= 1)

    #Constraint 2 - DST and RB1 cannot be on opposite teams in same game
    @variable(m, used_team_dst[i=1:num_teams], Bin)
    @variable(m, used_team_rb1[i=1:num_teams], Bin)
    @variable(m, same_game_rb1_dst, Bin)        
    @constraint(m, constr[i=1:num_teams, t=1:num_players], used_team_dst[i] >= players_teams[t, i]*player_lineup[t,9])
    @constraint(m, constr[i=1:num_teams, t=1:num_players], used_team_rb1[i] >= players_teams[t, i]*player_lineup[t,2])
    @constraint(m, constr[j=1:num_games], same_game_rb1_dst <= 2 - used_game_rb1[j]-used_game_dst[j])
    @constraint(m, constr[i=1:num_teams, t=1:num_players], used_team_rb1[i] >= used_team_dst[i]-same_game_rb1_dst)
    @constraint(m, sum(used_team_dst[i] for i=1:num_teams) <= 1)
    @constraint(m, sum(used_game_rb1[j] for j=1:num_games) <= 1)                                                                                                                                                                                                                                                 #Constraint 3 -  RB2 can be same game as WR3, no other players in this game
    @variable(m, used_game_rb2[i=1:num_games], Bin)
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_rb2[i] >= players_games[t, i]*player_lineup[t,1])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_rb2[i] >= players_games[t, i]*player_lineup[t,2])
    @constraint(m, constr[i=1:num_games, t=1:num_players], used_game_rb2[i] >= players_games[t, i]*player_lineup[t,3])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_rb2[i] >= players_games[t, i]*player_lineup[t,4])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_rb2[i] >= players_games[t, i]*player_lineup[t,5])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_rb2[i] >= players_games[t, i]*player_lineup[t,7])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_rb2[i] >= players_games[t, i]*player_lineup[t,8])
    @constraint(m, constr[i=1:num_games, t=1:num_players], 1-used_game_rb2[i] >= players_games[t, i]*player_lineup[t,9])
    @constraint(m, sum(used_game_rb2[i] for i=1:num_games) <= 1)

    #Constraint 4 -  WR3 not on same team as FLEX
    #@variable(m, used_team_wr3[i=1:num_teams], Bin)
    #@constraint(m, constr[i=1:num_teams, t=1:num_players], used_team_wr3[i] >= players_teams[t, i]*player_lineup[t,6])
    #@constraint(m, constr[i=1:num_teams, t=1:num_players], used_team_wr3[i] >= 1-players_teams[t, i]*player_lineup[t,8])
    #@constraint(m, sum(used_team_wr3[i] for i=1:num_teams) <= 1)



    # objective function
    @objective(m, Max, sum(player_info[i,:Points]*sum(player_lineup[i,j] for j=1:9) for i=1:num_players))

    # Solve the integer programming problem
    println("\tgenerating ", num_lineups, " lineups...")

    for w in 1:num_lineups
        println("\t",w)

        # add new overlap constraint
        if w > 1
            @constraint(m, sum(overlap[i,w-1]*sum(player_lineup[i,j] for j=1:9) for i=1:num_players) <= num_overlap)
        end
        # add new max appearances constraint
        if w > 1
            for i in 1:num_players
                @constraint(m, sum(overlap[i,k] for k=1:w-1)+sum(player_lineup[i,j] for j=1:9) <= max_appearances)
            end
        end
        status = solve(m)
        println("Objective value: ", getobjectivevalue(m))

        # Initialize a variable to store the total salary
        total_salary = 0

        # Calculate the total salary for the selected lineup
        for i in 1:num_players
            for j in 1:9
                if getvalue(player_lineup[i, j]) >= 0.99
                    total_salary += player_info[i, :Salary]
                end
            end
        end

        # Print the total salary
        println("Total Salary: ", total_salary)

        scores[w,1] = getobjectivevalue(m)
        scores[w,2] = total_salary

        # add lineup to lineup dataframe (including player ID for DK upload)
        for i in 1:num_players
            if sum(getvalue(player_lineup[i,j]) for j=1:9) >= 0.99
                overlap[i,w] = 1
                if getvalue(player_lineup[i,1]) >= 0.99
                    lineup[w,1] = player_info[i,:ID]
                elseif getvalue(player_lineup[i,2]) >= 0.99
                    lineup[w,2] = player_info[i,:ID]
                elseif getvalue(player_lineup[i,3]) >= 0.99
                    lineup[w,3] = player_info[i,:ID]
                elseif getvalue(player_lineup[i,4]) >= 0.99
                    lineup[w,4] = player_info[i,:ID]
                elseif getvalue(player_lineup[i,5]) >= 0.99
                    lineup[w,5] = player_info[i,:ID]
                elseif getvalue(player_lineup[i,6]) >= 0.99
                    lineup[w,6] = player_info[i,:ID]
                elseif getvalue(player_lineup[i,7]) >= 0.99
                    lineup[w,7] = player_info[i,:ID]
                elseif getvalue(player_lineup[i,8]) >= 0.99
                    lineup[w,8] = player_info[i,:ID]
                elseif getvalue(player_lineup[i,9]) >= 0.99
                    lineup[w,9] = player_info[i,:ID]
                end
            end
        end

        # add lineup to results dataframe (including player name for review after contests)
        for i in 1:num_players
            if sum(getvalue(player_lineup[i,j]) for j=1:9) >= 0.99
                overlap[i,w] = 1
                if getvalue(player_lineup[i,1]) >= 0.99
                    results[w,1] = player_info[i,:Name]
                elseif getvalue(player_lineup[i,2]) >= 0.99
                    results[w,2] = player_info[i,:Name]
                elseif getvalue(player_lineup[i,3]) >= 0.99
                    results[w,3] = player_info[i,:Name]
                elseif getvalue(player_lineup[i,4]) >= 0.99
                    results[w,4] = player_info[i,:Name]
                elseif getvalue(player_lineup[i,5]) >= 0.99
                    results[w,5] = player_info[i,:Name]
                elseif getvalue(player_lineup[i,6]) >= 0.99
                    results[w,6] = player_info[i,:Name]
                elseif getvalue(player_lineup[i,7]) >= 0.99
                    results[w,7] = player_info[i,:Name]
                elseif getvalue(player_lineup[i,8]) >= 0.99
                    results[w,8] = player_info[i,:Name]
                elseif getvalue(player_lineup[i,9]) >= 0.99
                    results[w,9] = player_info[i,:Name]
                end
            end
        end

        #Add in run parameters to results dataframe
        results[w,10] = string(points_system)
        results[w,11] = string(num_lineups)
        results[w,12] = string(num_overlap)
        results[w,13] = string(max_appearances)
        results[w,14] = string(alpha)
        results[w,15] = string(w)

    end
    results = hcat(results,scores)


    return lineup, results
end


function EM_algorithm(points_system, num_lineups, num_overlap, max_appearances, alpha, alpha_label, player_info, min_games, salary_cap, path_to_proj, date_string)
    # ===========================================================
    #                         RUNNING CODE
    # ===========================================================
    println("running models to generate lineups...")
    @printf("\n")
    lineups2, results2 = get_lineup(player_info, points_system, min_games, salary_cap, num_lineups, num_overlap, max_appearances, alpha)
    println("\tdone.")
    @printf("\n")

    println("finished running models.")
    @printf("\n")
    # ===========================================================
    #                    WRITE LINEUPS TO FILE
    # ===========================================================
    print("writing lineups to Lineups folder...")

    path_to_output = string(path_to_proj, "/Model/Output/Lineups_Draftkings/lineupsDK_",points_system,"_",alpha_label,"var_",num_overlap,"over_",max_appearances,"appear_",num_lineups,"lineups_", date_string,".csv")
    writetable(path_to_output,lineups2)
    path_to_output = string(path_to_proj, "/Model/Output/Lineups_Internal/lineups_",points_system,"_",alpha_label,"var_",num_overlap,"over_",max_appearances,"appear_",num_lineups,"lineups_", date_string,".csv")
    writetable(path_to_output,results2)

    println(" done.")
    @printf("\n")
end

function parse_args()
    date_list = eval(Meta.parse(ARGS[1]))
    points_system_list = eval(Meta.parse(ARGS[2]))
    num_lineups_list = eval(Meta.parse(ARGS[3]))
    num_overlap_list = eval(Meta.parse(ARGS[4]))
    max_appearances_list = eval(Meta.parse(ARGS[5]))
    alpha_list = eval(Meta.parse(ARGS[6]))
    path_to_proj = ARGS[7]
    return date_list, points_system_list, num_lineups_list, num_overlap_list, max_appearances_list, alpha_list, path_to_proj
end


if abspath(PROGRAM_FILE) == @__FILE__
    date_list, points_system_list, num_lineups_list, num_overlap_list, max_appearances_list, alpha_list, path_to_proj = parse_args()
    println("Parsed arguments:")
    println("date_list: ", date_list)
    println("points_system_list: ", points_system_list)
    println("num_lineups_list: ", num_lineups_list)
    println("num_overlap_list: ", num_overlap_list)
    println("max_appearances_list: ", max_appearances_list)
    println("alpha_list: ", alpha_list)
    println("path_to_proj: ", path_to_proj)
    
    main(date_list, points_system_list, num_lineups_list, num_overlap_list, max_appearances_list, alpha_list, path_to_proj)
end

