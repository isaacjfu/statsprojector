import { useState,useEffect } from "react"
const PlayerInfo = ( {playerData} ) => {
    useEffect ( () => {
        console.log("PLAYERDATA IN PLAYERINFO:" +playerData)
        console.log(playerData['image_url'])
        console.log(playerData['stats'])
        //console.log(playerData['stats'][0]['fg%'])
    }, [playerData])

    const handleClick = () => {
        
    }
  return (
    <div>
        <img src = {playerData.image_url} />
        { (!playerData || !playerData.stats) ? (
         (<></>)
        ) : (<table>
            <thead>
                <tr>
                    <th>Season</th>
                    <th>Team</th>
                    <th>Age</th>
                    <th>GP</th>
                    <th>MP</th>
                    <th>FG%</th>
                    <th>FT%</th>
                    <th>PTS</th>
                    <th>3PM</th>
                    <th>AST</th>
                    <th>REB</th>
                    <th>STL</th>
                    <th>BLK</th>
                    <th>TOV</th>
                </tr>
            </thead>
            <tbody>
                {playerData.stats.map((season) => {
                    return <tr>
                        <td>{season['season']}</td>
                        <td>{season['team']}</td>
                        <td>{season['age']}</td>
                        <td>{season['gp']}</td>
                        <td>{season['mp']}</td>
                        <td>{season['fg%']}</td>
                        <td>{season['ft%']}</td>
                        <td>{season['pts']}</td>
                        <td>{season['3p']}</td>
                        <td>{season['ast']}</td>
                        <td>{season['reb']}</td>
                        <td>{season['stl']}</td>
                        <td>{season['blk']}</td>
                        <td>{season['tov']}</td>
                    </tr>              
                    })}
            </tbody>
        </table>) 
        }
        <button onClick = {handleClick}> Project! </button>


    </div>
  )

}

export default PlayerInfo